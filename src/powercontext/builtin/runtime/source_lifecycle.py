from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select, update

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryCitation
from powercontext.builtin.artifacts.memory.errors import MemoryEntryNotFoundError
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.errors import RepositoryNotFoundError
from powercontext.builtin.persistence.sources import SourceRepository
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_TABLE
from powercontext.builtin.records import BaseValueNotFoundError
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources.models import SourceMaterialization, SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    pass


class SourceLifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(strict=True)
    revision: int = Field(strict=True, gt=0)
    sequence: int = Field(strict=True, gt=0)
    available: bool = Field(strict=True)
    reason: str = Field(strict=True)

    @field_validator("node_id", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            detail = "node_id and reason must not be blank"
            raise SourceLifecycleError(detail)
        return value


class _Node(BaseModel):
    kind: Literal["source", "memory", "topic"]
    reference: SourceRef | MemoryCitation | ArtifactRef
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] = "all"
    revision: int = 1
    event: SourceLifecycleEvent | None = None
    available: bool = True


class ScopedSourceLifecycle:
    def __init__(
        self, runtime: BuiltinRuntime, database: AsyncDatabase, sources: SourceRepository, scope_id: str
    ) -> None:
        self._runtime = runtime
        self._database = database
        self._sources = sources
        self.scope_id = validate_scope_id(scope_id)

    async def _load(self) -> dict[str, _Node]:
        async with self._database.transaction() as connection:
            raw = await connection.scalar(
                select(SOURCE_LIFECYCLE_TABLE.c.graph).where(SOURCE_LIFECYCLE_TABLE.c.scope_id == self.scope_id)
            )
        return {} if raw is None else {key: _Node.model_validate(value) for key, value in json.loads(raw).items()}

    async def _save(self, nodes: dict[str, _Node]) -> None:
        raw = json.dumps(
            {key: value.model_dump(mode="json") for key, value in nodes.items()},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        table = SOURCE_LIFECYCLE_TABLE
        async with self._database.transaction() as connection:
            exists = await connection.scalar(select(table.c.scope_id).where(table.c.scope_id == self.scope_id))
            if exists is None:
                await connection.execute(insert(table).values(scope_id=self.scope_id, graph=raw))
            else:
                await connection.execute(update(table).where(table.c.scope_id == self.scope_id).values(graph=raw))

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int = 1) -> None:
        if type(revision) is not int or revision <= 0:
            detail = "revision must be positive"
            raise SourceLifecycleError(detail)
        try:
            async with self._runtime._scoped_operation(self.scope_id), self._database.transaction() as connection:
                stored = await self._sources.get(connection, self.scope_id, source_ref)
        except RepositoryNotFoundError as error:
            detail = "source reference does not exist"
            raise SourceLifecycleError(detail) from error
        if stored.value.materialization != SourceMaterialization.CAPTURED:
            detail = "source reference must identify captured evidence"
            raise SourceLifecycleError(detail)
        await self._register(node_id, _Node(kind="source", reference=source_ref, revision=revision))

    async def register_memory(
        self, node_id: str, citation: MemoryCitation, *, parents: tuple[str, ...], policy: str
    ) -> None:
        try:
            await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        except (MemoryEntryNotFoundError, BaseValueNotFoundError) as error:
            detail = "memory reference does not exist"
            raise SourceLifecycleError(detail) from error
        if policy not in {"all", "any"}:
            detail = "unsupported dependency policy"
            raise SourceLifecycleError(detail)
        await self._register(
            node_id,
            _Node(kind="memory", reference=citation, parents=parents, policy=cast(Literal["all", "any"], policy)),
        )

    async def register_topic(
        self, node_id: str, reference: ArtifactRef, *, parents: tuple[str, ...], policy: str
    ) -> None:
        if reference.family != "topic-memory" or policy not in {"all", "any"}:
            detail = "invalid topic reference or dependency policy"
            raise SourceLifecycleError(detail)
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                reference.family, reference.artifact_id, reference.revision
            )
        except BaseValueNotFoundError as error:
            detail = "topic reference does not exist"
            raise SourceLifecycleError(detail) from error
        await self._register(
            node_id,
            _Node(kind="topic", reference=reference, parents=parents, policy=cast(Literal["all", "any"], policy)),
        )

    async def _register(self, node_id: str, node: _Node) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            detail = "node_id must not be blank"
            raise SourceLifecycleError(detail)
        async with self._runtime._scoped_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load()
            if node_id in nodes:
                prior = nodes[node_id].model_copy(update={"event": None, "available": True})
                if prior != node:
                    detail = "node registration conflicts with an existing binding"
                    raise SourceLifecycleError(detail)
                return
            if node.kind != "source" and (
                not node.parents
                or len(set(node.parents)) != len(node.parents)
                or any(parent not in nodes or parent == node_id for parent in node.parents)
            ):
                detail = "dependencies must be distinct previously registered nodes"
                raise SourceLifecycleError(detail)
            nodes[node_id] = node
            await self._propagate(nodes)
            await self._save(nodes)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        async with self._runtime._scoped_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load()
            node = nodes.get(event.node_id)
            if node is None or node.kind != "source" or node.revision != event.revision:
                detail = "event source or revision does not match"
                raise SourceLifecycleError(detail)
            if node.event is not None and (
                event.sequence < node.event.sequence or (event.sequence == node.event.sequence and event != node.event)
            ):
                detail = "event sequence is obsolete or conflicting"
                raise SourceLifecycleError(detail)
            node.event = event
            node.available = event.available
            await self._propagate(nodes)
            await self._save(nodes)
            return sorted(key for key, value in nodes.items() if value.available)

    async def _propagate(self, nodes: dict[str, _Node]) -> None:
        for node in nodes.values():
            if node.kind == "source":
                continue
            support = [nodes[parent].available for parent in node.parents]
            node.available = all(support) if node.policy == "all" else any(support)
        if not any(node.kind == "memory" for node in nodes.values()):
            return
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            for node in nodes.values():
                if node.kind != "memory":
                    continue
                citation = MemoryCitation.model_validate(node.reference.model_dump())
                entry = next(item for item in await service.entries(current) if item.entry_id == citation.entry_id)
                manifest = next(item for item in current.content.manifest.entries if item.entry_id == citation.entry_id)
                if (manifest.state == "active") != node.available:
                    current = await (
                        service.reactivate(current, entries=(entry,))
                        if node.available
                        else service.forget(current, entries=(entry,))
                    )

    async def resolve(self, node_id: str, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        node = (await self._load()).get(node_id)
        if node is None or purpose not in {"generation", "audit"}:
            detail = "unknown node or resolution purpose"
            raise SourceLifecycleError(detail)
        if purpose == "generation" and not node.available:
            detail = "withdrawn evidence cannot be used for generation"
            raise SourceLifecycleError(detail)
        return node.reference

    async def unavailable_topics(self) -> frozenset[str]:
        return frozenset(
            node.reference.artifact_id
            for node in (await self._load()).values()
            if node.kind == "topic" and not node.available and isinstance(node.reference, ArtifactRef)
        )


class SourceLifecycleApplication:
    def __init__(
        self, runtime: BuiltinRuntime, database: AsyncDatabase | None, sources: SourceRepository | None
    ) -> None:
        self._runtime = runtime
        self._database = database
        self._sources = sources

    def for_scope(self, scope_id: str) -> ScopedSourceLifecycle:
        if self._database is None or self._sources is None:
            detail = "source lifecycle requires relational persistence"
            raise SourceLifecycleError(detail)
        return ScopedSourceLifecycle(self._runtime, self._database, self._sources, scope_id)
