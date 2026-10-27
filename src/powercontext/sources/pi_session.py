# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture the raw visible events of one Pi session branch as durable Sources."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import cast

import rfc8785
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from powercontext.limits import MAX_SOURCE_ID_LENGTH, MAX_SOURCE_OBSERVATION_BYTES
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinitionRegistry
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.observations import project_source_for_transport
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_CONTRACT_VERSION = "1"
PI_EVENT_ROLE_CUSTOM = "custom"
_JSON_VALUE = TypeAdapter(JsonValue)


class _InvalidPiSessionError(ValueError):
    pass


class PiSessionInput(BaseModel):
    """One complete Pi v3 session archive and the branch leaf to capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event from the selected branch with its original record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured Pi event that keeps the full original record."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build a fixed-length identity that cannot collide across sessions."""

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")  # noqa: TRY003
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")  # noqa: TRY003
    payload = json.dumps([session_id, event_id], ensure_ascii=False, separators=(",", ":"))
    identity = f"{PI_EVENT_SOURCE_NAME}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"pi event identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Return the visible events on the selected leaf's ancestor chain, root first."""

    records = _read_archive(value.content)
    session_id, events = _build_branch(records, value.leaf_id)
    return tuple(
        PiEventInput(
            session_id=session_id,
            event_id=event_id,
            parent_id=parent_id,
            event_type=event_type,
            role=role,
            content=content,
            record=record,
        )
        for event_id, parent_id, event_type, role, content, record in events
    )


def _read_archive(content: bytes) -> list[dict[str, JsonValue]]:
    if not isinstance(content, bytes):
        raise TypeError("pi session content must be bytes")  # noqa: TRY003
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        detail = "pi session must be valid utf-8"
        raise _InvalidPiSessionError(detail) from error
    records: list[dict[str, JsonValue]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(
                line,
                parse_constant=_reject_json_constant,
                parse_float=_parse_float,
                object_pairs_hook=_json_object,
            )
            _validate_json_value({"record": parsed})
        except (ValueError, RecursionError) as error:
            detail = f"pi session line {number} is not valid supported JSON"
            raise _InvalidPiSessionError(detail) from error
        if not isinstance(parsed, dict):
            detail = f"pi session line {number} must be a json object"
            raise _InvalidPiSessionError(detail)
        records.append(parsed)
    if not records:
        raise ValueError("pi session archive must not be empty")  # noqa: TRY003
    return records


def _reject_json_constant(value: str) -> None:
    detail = f"invalid JSON constant: {value}"
    raise _InvalidPiSessionError(detail)


def _parse_float(value: str) -> float:
    number = float(value)
    mantissa = value.lower().partition("e")[0]
    if not math.isfinite(number) or (number == 0 and any(digit in "123456789" for digit in mantissa)):
        detail = "pi JSON number is outside the supported floating-point range"
        raise _InvalidPiSessionError(detail)
    return number


def _json_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            detail = "pi JSON object contains a duplicate key"
            raise _InvalidPiSessionError(detail)
        result[key] = value
    return result


def _validate_json_value(value: JsonValue) -> None:
    try:
        encoded = rfc8785.dumps({"value": {"payload": value}})
        rfc8785.dumps(_JSON_VALUE.validate_json(encoded, strict=True))
    except (ValueError, TypeError, RecursionError) as error:
        detail = "pi JSON value cannot round-trip through Source storage"
        raise _InvalidPiSessionError(detail) from error


def _build_branch(
    records: list[dict[str, JsonValue]],
    leaf_id: str | None,
) -> tuple[str, list[tuple[str, str | None, str, str, str, dict[str, JsonValue]]]]:
    header = records[0]
    if header.get("type") != "session" or type(header.get("version")) is not int or header["version"] != 3:
        raise ValueError("pi session header must declare type=session and version=3")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("pi session header must declare a non-empty id")  # noqa: TRY003

    parents, by_id = _event_graph(records[1:])
    if leaf_id is not None and leaf_id not in by_id:
        detail = f"pi session leaf {leaf_id!r} does not exist"
        raise ValueError(detail)
    leaf = leaf_id if leaf_id is not None else next(reversed(by_id), None)
    chain: list[str] = []
    cursor = leaf
    while cursor is not None:
        chain.append(cursor)
        cursor = parents[cursor]
    events: list[tuple[str, str | None, str, str, str, dict[str, JsonValue]]] = []
    for event_id in reversed(chain):
        record = by_id[event_id]
        event_type = cast(str, record["type"])
        rendered = _render_event(record, event_type)
        if rendered is not None:
            role, body = rendered
            events.append((event_id, parents[event_id], event_type, role, f"{role}: {body}", record))
    return session_id, events


def _event_graph(
    records: list[dict[str, JsonValue]],
) -> tuple[dict[str, str | None], dict[str, dict[str, JsonValue]]]:
    parents: dict[str, str | None] = {}
    by_id: dict[str, dict[str, JsonValue]] = {}
    for record in records:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("every pi record must declare a non-empty id")  # noqa: TRY003
        if event_id in by_id:
            raise ValueError(f"duplicate pi record id {event_id!r}")  # noqa: TRY003
        event_type = record.get("type")
        if not isinstance(event_type, str) or not event_type.strip() or event_type == "session":
            detail = "pi event must declare a non-session type"
            raise _InvalidPiSessionError(detail)
        if "parentId" not in record:
            detail = "pi event must declare parentId"
            raise _InvalidPiSessionError(detail)
        parent_id = record.get("parentId")
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in by_id):
            raise ValueError(f"pi record {event_id!r} references unknown parent {parent_id!r}")  # noqa: TRY003
        by_id[event_id] = record
        parents[event_id] = parent_id
    return parents, by_id


def _render_event(record: dict[str, JsonValue], event_type: str) -> tuple[str, str] | None:
    if event_type == "compaction" or event_type == "branch_summary":
        summary = record.get("summary")
        if not isinstance(summary, str):
            raise ValueError(f"pi {event_type} record must declare a summary")  # noqa: TRY003
        return (event_type, summary) if summary else None
    if event_type == "custom_message":
        body = _visible_text(record.get("content"))
        if body is None:
            return None
        return PI_EVENT_ROLE_CUSTOM, body
    if event_type != "message":
        return None

    message = record.get("message")
    if not isinstance(message, dict):
        detail = "pi message record must declare a message object"
        raise _InvalidPiSessionError(detail)
    return _render_message(message)


def _render_message(message: dict[str, JsonValue]) -> tuple[str, str] | None:
    role = _require_text(message.get("role"))
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output", "")
        if not isinstance(command, str) or not isinstance(output, str):
            detail = "pi bash command and output must be strings"
            raise _InvalidPiSessionError(detail)
        return role, f"$ {command}\n{output}"
    if role == "toolResult":
        tool_name = message.get("toolName")
        tool_call_id = message.get("toolCallId")
        body = _visible_text(message.get("content")) or ""
        if not body:
            return None
        return role, f"[tool_result {_require_text(tool_name)} {_require_text(tool_call_id)}] {body}"
    body = _render_message_content(message.get("content"))
    if body is None:
        return None
    return role, body


def _render_message_content(content: object) -> str | None:
    if isinstance(content, str):
        return content if content else None
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif kind == "toolCall":
            arguments = block.get("arguments")
            encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            parts.append(f"[tool_call {_require_text(block.get('name'))} {_require_text(block.get('id'))}] {encoded}")
    return "\n".join(parts) if any(parts) else None


def _visible_text(content: object) -> str | None:
    if isinstance(content, str):
        return content if content else None
    if not isinstance(content, list):
        return None
    parts = [block.get("text") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    texts = [text for text in parts if isinstance(text, str)]
    return "\n".join(texts) if any(texts) else None


def _require_text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("pi tool metadata must be a non-empty string")  # noqa: TRY003
    return value


class PiEventSourceAdapter:
    """Resolve one parsed event into a captured Source and read it back."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        _validate_json_value(value.model_dump())
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            definition_version=PI_CONTRACT_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=deepcopy(value.record),
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=deepcopy(source.record),
        )


class PiEventTextEvidenceProjection:
    """Expose the complete event text with its identity metadata."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=PI_CONTRACT_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


class _PiCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    session_id: str
    leaf_id: str | None
    imported: dict[str, str]


class PiSessionConnector:
    """Import only the branch events a persisted checkpoint has not seen."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_CONTRACT_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: Path | str, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = await asyncio.to_thread(self.path.read_bytes)
        records = _read_archive(raw)
        session_id, branch = _build_branch(records, self.leaf_id)
        events = tuple(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=parent_id,
                event_type=event_type,
                role=role,
                content=content,
                record=record,
            )
            for event_id, parent_id, event_type, role, content, record in branch
        )
        by_id = {cast(str, record["id"]): record for record in records[1:]}
        imported: dict[str, str] = {}
        if session.checkpoint is not None:
            previous = _PiCheckpoint.model_validate(session.checkpoint)
            if previous.version != PI_CONTRACT_VERSION or previous.session_id != session_id:
                detail = "pi checkpoint version or session does not match"
                raise _InvalidPiSessionError(detail)
            for event_id, digest in previous.imported.items():
                record = by_id.get(event_id)
                if record is None or _fingerprint(record) != digest:
                    detail = "previously imported pi event was changed or removed"
                    raise _InvalidPiSessionError(detail)
            imported.update(previous.imported)
        pending = tuple((event, _fingerprint(event.record)) for event in events if event.event_id not in imported)
        registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        for event, _ in pending:
            source = await registry.resolve(event)
            observation = project_source_for_transport(registry, source)
            size = len(observation.model_dump_json().encode("utf-8"))
            if size > MAX_SOURCE_OBSERVATION_BYTES:
                detail = f"pi observation is {size} bytes; limit is {MAX_SOURCE_OBSERVATION_BYTES} bytes"
                raise _InvalidPiSessionError(detail)
        for event, digest in pending:
            await session.submit(pi_event_source_id(session_id, event.event_id), PI_EVENT_SOURCE_NAME, event)
            imported[event.event_id] = digest

        checkpoint: JsonValue = {
            "version": PI_CONTRACT_VERSION,
            "session_id": session_id,
            "leaf_id": self.leaf_id if self.leaf_id is not None else (records[-1]["id"] if len(records) > 1 else None),
            "imported": imported,
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


def _fingerprint(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "PI_CONTRACT_VERSION",
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
