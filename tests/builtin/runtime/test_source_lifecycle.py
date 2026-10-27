from __future__ import annotations

import asyncio

import pytest

from powercontext.builtin.artifacts.memory import MemoryEntryInput, MemoryService
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    SearchMemoryRequest,
    SubmitSourceObservation,
    open_builtin_contexts,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleError, SourceLifecycleEvent
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import BUILTIN_SOURCE_REGISTRY
from powercontext.sources import (
    SourceDefinitionRegistry,
    SourceRef,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import CSV_ROW_SOURCE_DEFINITION, CSVDocumentInput, parse_csv_document
from powercontext.sources.pdf_pages import PDF_PAGE_SOURCE_DEFINITION, PDFPageInput
from powercontext.sources.pi_session import PI_EVENT_SOURCE_DEFINITION, PiSessionInput, parse_pi_session


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("failure_type", [OSError, asyncio.CancelledError])
async def test_partial_native_state_write_retries_after_restart(tmp_path, monkeypatch, available, failure_type):
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'lifecycle.db'}"))
    citations = []
    async with open_builtin_runtime(config) as runtime:
        scope = await runtime.scopes.create(ScopeDraft(title="Lifecycle", summary="Retry", idempotency_key="retry"))
        memory = runtime.memory.for_scope(scope.scope_id)
        service = runtime.lifecycle.for_scope(scope.scope_id)
        source = await runtime.sources.for_scope(scope.scope_id).capture(
            CaptureSource(source_id="origin", content="Original evidence", metadata={})
        )
        await service.register_source("origin", source.source_ref)
        for index in range(2):
            result = await memory.remember(
                RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=f"retryneedle{index} evidence"),))
            )
            citations.append(result.entry.citation)
            await service.register_memory(f"derived-{index}", result.entry.citation, parents=("origin",), policy="all")
        if available:
            await service.apply(
                SourceLifecycleEvent(node_id="origin", revision=1, sequence=1, available=False, reason="withdraw")
            )
        event = SourceLifecycleEvent(
            node_id="origin", revision=1, sequence=2 if available else 1, available=available, reason="retry"
        )
        before = (await memory.list(include_inactive=True)).memory_ref
        operation = "reactivate" if available else "forget"
        original = getattr(MemoryService, operation)
        attempted = 0

        async def interrupted(self, current, *, entries, reason=None):
            nonlocal attempted
            attempted += 1
            if attempted == 2:
                detail = "interrupted state write"
                raise failure_type(detail)
            return await original(self, current, entries=entries, reason=reason)

        with monkeypatch.context() as patch:
            patch.setattr(MemoryService, operation, interrupted)
            with pytest.raises(failure_type):
                await service.apply(event)
        partial = await memory.list(include_inactive=True)
        assert {entry.state for entry in partial.entries} == {"active", "inactive"}
        assert partial.memory_ref.revision == before.revision + 1

    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope.scope_id)
        service = runtime.lifecycle.for_scope(scope.scope_id)
        active = await service.apply(event)
        assert active == (["derived-0", "derived-1", "origin"] if available else [])
        completed = await memory.list(include_inactive=True)
        assert completed.memory_ref.revision == before.revision + 2
        assert {entry.state for entry in completed.entries} == ({"active"} if available else {"inactive"})
        assert await service.apply(event) == active
        assert (await memory.list(include_inactive=True)).memory_ref == completed.memory_ref
        for index, citation in enumerate(citations):
            node_id = f"derived-{index}"
            assert await service.resolve(node_id, purpose="audit") == citation
            if available:
                assert await service.resolve(node_id, purpose="generation") == citation
            else:
                with pytest.raises(SourceLifecycleError):
                    await service.resolve(node_id, purpose="generation")
            original_entry = await memory.get(GetMemoryEntryRequest(citation=citation))
            assert original_entry.entry.text == f"retryneedle{index} evidence"
            hits = await memory.search(SearchMemoryRequest(query=f"retryneedle{index}", mode="fts", limit=10))
            assert bool(hits.hits) is available


@pytest.mark.parametrize("source_type", ["pi", "csv", "pdf"])
@pytest.mark.parametrize("capture_mode", ["observation", "native"])
async def test_captured_source_types_register_propagate_and_reopen(tmp_path, source_type, capture_mode):
    if source_type == "pi":
        definition = PI_EVENT_SOURCE_DEFINITION
        value = parse_pi_session(
            PiSessionInput(
                content=b"\n".join([
                    b'{"type":"session","version":3,"id":"audit"}',
                    b'{"type":"message","id":"event","parentId":null,"message":{"role":"user","content":"evidence"}}',
                ])
            )
        )[0]
    elif source_type == "csv":
        definition = CSV_ROW_SOURCE_DEFINITION
        value = parse_csv_document(
            CSVDocumentInput(document_id="audit", primary_key="id", content=b"id,text\na,evidence")
        )[0]
    else:
        definition = PDF_PAGE_SOURCE_DEFINITION
        value = PDFPageInput(document_id="audit", document_sha256="a" * 64, page_number=1, text="evidence")
    registry = SourceDefinitionRegistry((*BUILTIN_SOURCE_REGISTRY.definitions, definition))
    runtime_registry = registry if capture_mode == "native" else BUILTIN_SOURCE_REGISTRY
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'cross-source.db'}"))
    source = await registry.resolve(value)
    reference = SourceRef(source_type=definition.name, source_id=source.name)
    async with open_builtin_runtime(config, source_registry=runtime_registry) as runtime:
        scope = await runtime.scopes.create(ScopeDraft(title="Sources", summary="Lifecycle", idempotency_key="source"))
        other = await runtime.scopes.create(ScopeDraft(title="Other", summary="Isolation", idempotency_key="other"))
        if capture_mode == "native":
            async with open_builtin_contexts(config, source_registry=registry) as contexts:
                await (await contexts.get(scope.scope_id)).sources.add(source)
        else:
            await runtime.ingestion.register(manifest_for_definition(definition))
            await runtime.ingestion.submit(
                SubmitSourceObservation(
                    scope_id=scope.scope_id, observation=project_source_for_transport(registry, source)
                )
            )
        service = runtime.lifecycle.for_scope(scope.scope_id)
        await service.register_source("origin", reference)
        await service.register_source("origin", reference)
        with pytest.raises(SourceLifecycleError):
            await runtime.lifecycle.for_scope(other.scope_id).register_source("origin", reference)
        with pytest.raises(SourceLifecycleError):
            await service.register_source("absent", SourceRef(source_type=definition.name, source_id="absent"))
        memory = runtime.memory.for_scope(scope.scope_id)
        result = await memory.remember(
            RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text="crossneedle evidence"),))
        )
        citation = result.entry.citation
        await service.register_memory("derived", citation, parents=("origin",), policy="all")
        assert (
            await service.apply(
                SourceLifecycleEvent(node_id="origin", revision=1, sequence=1, available=False, reason="withdraw")
            )
            == []
        )
        assert not (await memory.search(SearchMemoryRequest(query="crossneedle", mode="fts"))).hits
    async with open_builtin_runtime(config, source_registry=runtime_registry) as runtime:
        service = runtime.lifecycle.for_scope(scope.scope_id)
        assert await service.resolve("origin", purpose="audit") == reference
        with pytest.raises(SourceLifecycleError):
            await service.resolve("derived", purpose="generation")
        assert await service.apply(
            SourceLifecycleEvent(node_id="origin", revision=1, sequence=2, available=True, reason="restore")
        ) == ["derived", "origin"]
        assert await service.resolve("derived", purpose="generation") == citation
        memory = runtime.memory.for_scope(scope.scope_id)
        assert (await memory.get(GetMemoryEntryRequest(citation=citation))).entry.text == "crossneedle evidence"
        assert (await memory.search(SearchMemoryRequest(query="crossneedle", mode="fts"))).hits
