from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import pytest

from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CommitConnectorCheckpoint,
    SubmitSourceObservation,
    open_builtin_contexts,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.errors import ConnectorSubmissionRejectedError, SourceConflictError
from powercontext.limits import MAX_SOURCE_OBSERVATION_BYTES
from powercontext.sources import (
    ConnectorBinding,
    ConnectorRunStatus,
    SourceDefinitionRegistry,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
)


class _RuntimeSink:
    def __init__(self, runtime, registry):
        self.runtime = runtime
        self.registry = registry

    async def submit(self, binding, item_id, definition_name, value):
        source = await self.registry.resolve(value)
        observation = project_source_for_transport(self.registry, source)
        return (
            await self.runtime.ingestion.submit(
                SubmitSourceObservation(scope_id=binding.scope_id, observation=observation)
            )
        ).source_ref


class _RuntimeCheckpoints:
    def __init__(self, runtime):
        self.runtime = runtime

    async def load(self, binding):
        return (await self.runtime.ingestion.checkpoint(binding)).checkpoint

    async def save(self, binding, checkpoint, *, expected):
        await self.runtime.ingestion.commit(
            CommitConnectorCheckpoint(binding=binding, checkpoint=checkpoint, expected=expected)
        )


async def _run_persisted(directory, connector, binding, *, failure=None):
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{directory / 'connector.db'}"))
    async with open_builtin_runtime(config) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Pi", summary="Evidence", idempotency_key=binding.scope_id)
        )
        binding = binding.model_copy(update={"scope_id": scope.scope_id})
        await runtime.ingestion.register(manifest_for_definition(PI_EVENT_SOURCE_DEFINITION))
        sink = _RuntimeSink(runtime, registry)

        class InterruptingSink:
            async def submit(self, binding, item_id, definition_name, value):
                if failure is not None and value.event_id == "left":
                    raise failure
                return await sink.submit(binding, item_id, definition_name, value)

        result = await ConnectorLifecycle(sink=InterruptingSink(), checkpoints=_RuntimeCheckpoints(runtime)).run(
            connector, binding
        )
    async with open_builtin_contexts(config) as contexts:
        entries = await (await contexts.get(binding.scope_id)).sources.journal.entries()
    return result, entries


def _archive(records, session_id="session"):
    return "\n".join(
        json.dumps(row, ensure_ascii=False) for row in [{"type": "session", "version": 3, "id": session_id}, *records]
    ).encode()


@pytest.mark.parametrize(
    "record",
    [
        {"type": "model_change", "id": "a"},
        {"type": "model_change", "id": "a", "parentId": []},
        {"type": "model_change", "id": "a", "parentId": "later"},
        {"type": "session", "id": "nested", "parentId": None},
    ],
)
def test_invalid_records_fail(record):
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=_archive([record])))


async def test_full_unicode_tool_record_and_bounded_identity():
    text = "Full tool output é\u2028line\n" * 3_000
    record = {
        "type": "message",
        "id": "event" * 100,
        "parentId": None,
        "message": {"role": "toolResult", "toolName": "read", "toolCallId": "call", "content": text},
        "details": {"exitCode": 0},
    }
    event = parse_pi_session(PiSessionInput(content=_archive([record], "session" * 100)))[0]
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    source = await registry.resolve(event)
    assert len(source.name) <= 256
    assert await registry.read(source) == event
    observation = project_source_for_transport(registry, source)
    assert observation.projections[0].value["content"] == f"toolResult: [tool_result read call] {text}"
    restored = await registry.read(source)
    restored.record["changed"] = True
    assert "changed" not in (await registry.read(source)).record


async def test_identity_uses_only_session_and_event_without_delimiter_ambiguity():
    record = {"type": "message", "id": "b:c", "parentId": None, "message": {"role": "user", "content": "first"}}
    event = parse_pi_session(PiSessionInput(content=_archive([record], "a")))[0]
    other = event.model_copy(update={"session_id": "a:b", "event_id": "c"})
    changed = event.model_copy(update={"content": "different", "parent_id": "changed", "record": {"changed": True}})
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    original = await registry.resolve(event)
    assert (await registry.resolve(changed)).name == original.name
    assert (await registry.resolve(other)).name != original.name


def test_invalid_unselected_branch_still_rejects_the_complete_archive():
    records = [
        {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "root"}},
        {"type": "message", "id": "bad", "parentId": "absent", "message": {"role": "user", "content": "bad"}},
    ]
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=_archive(records), leaf_id="root"))


async def test_format_only_rewrites_and_empty_session_append_resume(tmp_path):
    path = tmp_path / "session.jsonl"
    binding = ConnectorBinding(
        scope_id="pi", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )
    path.write_bytes(_archive([]))
    empty, _ = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert empty.items == () and empty.committed_checkpoint is not None
    root = {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "root"}}
    path.write_bytes(_archive([root]))
    first, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert len(first.items) == len(entries) == 1
    rewritten = "\r\n".join(json.dumps(json.loads(line), sort_keys=True) for line in path.read_text().splitlines())
    path.write_bytes(b"\xef\xbb\xbf" + rewritten.encode())
    replay, _ = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert replay.items == () and replay.committed_checkpoint == first.committed_checkpoint


async def test_cross_branch_metadata_changes_fail_before_import_and_recover(tmp_path: Path):
    path = tmp_path / "session.jsonl"
    records = [
        {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "shared"}},
        {"type": "message", "id": "left", "parentId": "root", "message": {"role": "assistant", "content": "left"}},
        {"type": "message", "id": "right", "parentId": "root", "message": {"role": "assistant", "content": "right"}},
    ]
    binding = ConnectorBinding(
        scope_id="pi", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )

    async def run(leaf):
        return await _run_persisted(tmp_path, PiSessionConnector(path, leaf), binding)

    path.write_bytes(_archive(records))
    first, _ = await run("left")
    records[1]["metadata"] = "changed while on another branch"
    path.write_bytes(_archive(records))
    with pytest.raises(ValueError):
        await run("right")
    del records[1]["metadata"]
    path.write_bytes(_archive(records))
    recovered, entries = await run("right")
    assert recovered.previous_checkpoint == first.committed_checkpoint
    assert len(recovered.items) == 1 and len(entries) == 3
    replay, _ = await run("right")
    assert replay.items == ()


@pytest.mark.parametrize(
    "failure", [OSError("interrupted"), ConnectorSubmissionRejectedError("rejected"), asyncio.CancelledError()]
)
async def test_failed_partial_import_does_not_advance_checkpoint_and_replays_idempotently(tmp_path, failure):
    path = tmp_path / "session.jsonl"
    records = [
        {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "root"}},
        {"type": "message", "id": "middle", "parentId": "root", "message": {"role": "assistant", "content": "middle"}},
        {"type": "message", "id": "left", "parentId": "middle", "message": {"role": "assistant", "content": "left"}},
    ]
    path.write_bytes(_archive(records))
    binding = ConnectorBinding(
        scope_id="pi", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )
    first, _ = await _run_persisted(tmp_path, PiSessionConnector(path, "root"), binding)
    before = deepcopy(first.committed_checkpoint)
    if isinstance(failure, ConnectorSubmissionRejectedError):
        interrupted, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding, failure=failure)
        assert interrupted.status is ConnectorRunStatus.INCOMPLETE
        assert interrupted.committed_checkpoint == before
        assert len(entries) == 2
    else:
        with pytest.raises(type(failure)):
            await _run_persisted(tmp_path, PiSessionConnector(path), binding, failure=failure)
    recovered, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert recovered.previous_checkpoint == before
    assert len(recovered.items) == 2 and len(entries) == 3
    replay, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert replay.items == () and len(entries) == 3


@pytest.mark.parametrize("change", ["remove", "session", "malformed-tail"])
async def test_changed_archive_rejects_before_new_event_and_preserves_progress(tmp_path, change):
    path = tmp_path / "session.jsonl"
    root = {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "root"}}
    tail = {"type": "message", "id": "tail", "parentId": "root", "message": {"role": "assistant", "content": "tail"}}
    path.write_bytes(_archive([root]))
    binding = ConnectorBinding(
        scope_id="pi", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )
    first, _ = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    if change == "remove":
        path.write_bytes(_archive([]))
    elif change == "session":
        path.write_bytes(_archive([root, tail], "changed"))
    else:
        path.write_bytes(_archive([root, tail]) + b"{bad json}\n")
    with pytest.raises(ValueError):
        await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    path.write_bytes(_archive([root, tail]))
    recovered, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert recovered.previous_checkpoint == first.committed_checkpoint
    assert len(recovered.items) == 1 and len(entries) == 2


async def test_large_tool_source_is_persisted_and_same_text_metadata_conflicts(tmp_path):
    path = tmp_path / "session.jsonl"
    text = "full tool output é\n" * 3_000
    record = {
        "type": "message",
        "id": "tool",
        "parentId": None,
        "message": {"role": "toolResult", "toolName": "read", "toolCallId": "call", "content": text},
        "details": {"exitCode": 0},
    }
    path.write_bytes(_archive([record]))
    binding = ConnectorBinding(
        scope_id="pi", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )
    result, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert len(entries) == 1
    observation = entries[0].source
    assert observation.projections[0].value["content"] == f"toolResult: [tool_result read call] {text}"
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    restored = PI_EVENT_SOURCE_DEFINITION.source_class.model_validate(observation.payload)
    assert (await registry.read(restored)).record == record
    changed = deepcopy(record)
    changed["details"] = {"exitCode": 1}
    event = parse_pi_session(PiSessionInput(content=_archive([changed])))[0]
    conflict = project_source_for_transport(registry, await registry.resolve(event))
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'connector.db'}"))
    async with open_builtin_runtime(config) as runtime:
        with pytest.raises(SourceConflictError):
            await runtime.ingestion.submit(
                SubmitSourceObservation(scope_id=result.binding.scope_id, observation=conflict)
            )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"type":"session","version":3.0,"id":"s"}',
        b'{"type":"session","version":true,"id":"s"}',
        b'{"type":"session","version":3,"id":"   "}',
        _archive([]) + b'{"type":"model_change","id":"a","parentId":null,"value":NaN}',
    ],
)
def test_invalid_header_and_non_json_numbers(raw):
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=raw))


def test_deep_chain_and_invisible_events_do_not_discard_original_evidence():
    records = [{"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "original"}}]
    parent = "root"
    for index in range(3_000):
        node = f"control-{index}"
        records.append({"type": "model_change", "id": node, "parentId": parent})
        parent = node
    records.extend([
        {
            "type": "message",
            "id": "hidden",
            "parentId": parent,
            "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hidden"}]},
        },
        {"type": "compaction", "id": "summary", "parentId": "hidden", "summary": "retained summary"},
        {
            "type": "custom_message",
            "id": "custom",
            "parentId": "summary",
            "content": "retained custom",
            "display": False,
        },
    ])
    events = parse_pi_session(PiSessionInput(content=_archive(records)))
    assert [event.event_id for event in events] == ["root", "summary", "custom"]
    assert events[1].parent_id == "hidden"
    assert parse_pi_session(PiSessionInput(content=_archive([]))) == ()


@pytest.mark.parametrize(
    "raw_record",
    [
        b'{"type":"message","id":"bad","parentId":null,"message":{"role":"user","content":"\\ud800"}}',
        b'{"type":"model_change","id":"bad","parentId":null,"details":{"\\udfff":"value"}}',
        b'{"type":"model_change","id":"bad","parentId":null,"details":{"value":1e400}}',
        b'{"type":"model_change","id":"bad","parentId":null,"details":{"value":-1e400}}',
        b'{"type":"model_change","id":"bad","parentId":null,"details":{"value":1e-400}}',
        b'{"type":"model_change","id":"bad","parentId":null,"details":{"value":1e-999999999999999999999999999}}',
        b'{"type":"model_change","id":"bad","parentId":null,"details":{"value":9007199254740992}}',
        b'{"type":"model_change","id":"bad","parentId":null,"details":{"value":1e18}}',
        b'{"type":"model_change","id":"bad","parentId":null,"details":{"value":0,"value":1}}',
        b'{"type":"model_change","id":"bad","parentId":null,"details":' + b"[" * 197 + b"0" + b"]" * 197 + b"}",
        b'{"type":"model_change","id":"bad","parentId":null,"details":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}",
    ],
)
def test_invalid_json_values_fail_even_outside_selected_branch(raw_record):
    root = {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "root"}}
    raw = _archive([root]) + b"\n" + raw_record
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=raw, leaf_id="root"))


@pytest.mark.parametrize("number", [b"0e999999999999999999999999999", b"-0e-999999999999999999999999999"])
def test_zero_with_extreme_exponents_remains_supported(number):
    raw_record = (
        b'{"type":"message","id":"zero","parentId":null,'
        b'"message":{"role":"user","content":"zero"},"value":' + number + b"}"
    )
    event = parse_pi_session(PiSessionInput(content=_archive([]) + b"\n" + raw_record))[0]
    assert event.record["value"] == 0


async def test_unicode_pairs_and_supported_numeric_boundaries_persist(tmp_path):
    text = "Unicode \U0001f600 and \u2028 remain intact"
    record = {
        "type": "message",
        "id": "numbers",
        "parentId": None,
        "message": {"role": "user", "content": text},
        "details": {"values": [9007199254740991, -9007199254740991, 1e100, 5e-324, -0.0, 1.5, True, None]},
    }
    raw = b"\n".join(json.dumps(row).encode() for row in [{"type": "session", "version": 3, "id": "session"}, record])
    path = tmp_path / "session.jsonl"
    path.write_bytes(raw)
    binding = ConnectorBinding(
        scope_id="pi", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )
    first, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert len(entries) == 1
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    source = PI_EVENT_SOURCE_DEFINITION.source_class.model_validate(entries[0].source.payload)
    restored = await registry.read(source)
    assert restored.record == record
    assert restored.content == f"user: {text}"
    replay, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert replay.items == () and replay.committed_checkpoint == first.committed_checkpoint
    assert len(entries) == 1


@pytest.mark.parametrize("invalid", [float("inf"), float("nan"), 9007199254740992, {"nested": "\ud800"}])
async def test_adapter_rejects_unstorable_record_before_projection(invalid):
    record = {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "root"}}
    event = parse_pi_session(PiSessionInput(content=_archive([record])))[0]
    event.record["details"] = invalid
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    with pytest.raises(ValueError):
        await registry.resolve(event)


async def test_nested_json_storage_boundary_persists_and_recovers(tmp_path):
    path = tmp_path / "session.jsonl"
    root = {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "root"}}
    pending = {
        "type": "message",
        "id": "pending",
        "parentId": "root",
        "message": {"role": "user", "content": "pending"},
    }
    nested = 0
    for _ in range(196):
        nested = [nested]
    tail = {
        "type": "message",
        "id": "tail",
        "parentId": "pending",
        "message": {"role": "user", "content": "nested record"},
        "details": nested,
    }
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    event = parse_pi_session(PiSessionInput(content=_archive([root, pending, tail])))[-1]
    event.record["details"] = [nested]
    with pytest.raises(ValueError):
        await registry.resolve(event)
    binding = ConnectorBinding(
        scope_id="pi", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )
    path.write_bytes(_archive([root]))
    first, _ = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    tail["details"] = [nested]
    path.write_bytes(_archive([root, pending, tail]))
    with pytest.raises(ValueError):
        await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'connector.db'}"))
    async with open_builtin_contexts(config) as contexts:
        assert len(await (await contexts.get(first.binding.scope_id)).sources.journal.entries()) == 1
        assert (await contexts.connector_checkpoint(first.binding)).checkpoint == first.committed_checkpoint
    tail["details"] = nested
    path.write_bytes(_archive([root, pending, tail]))
    recovered, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert recovered.previous_checkpoint == first.committed_checkpoint
    assert len(recovered.items) == 2 and len(entries) == 3
    source = PI_EVENT_SOURCE_DEFINITION.source_class.model_validate(entries[-1].source.payload)
    assert (await registry.read(source)).record == tail
    replay, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert replay.items == () and len(entries) == 3


async def test_invalid_tail_preserves_checkpoint_and_leaves_no_partial_sources(tmp_path):
    path = tmp_path / "session.jsonl"
    root = {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "root"}}
    pending = {
        "type": "message",
        "id": "pending",
        "parentId": "root",
        "message": {"role": "user", "content": "pending"},
    }
    tail = {"type": "message", "id": "tail", "parentId": "pending", "message": {"role": "user", "content": "tail"}}
    binding = ConnectorBinding(
        scope_id="pi", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )
    path.write_bytes(_archive([root]))
    first, _ = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    tail["details"] = "\ud800"
    path.write_bytes(
        b"\n".join(
            json.dumps(row).encode()
            for row in [{"type": "session", "version": 3, "id": "session"}, root, pending, tail]
        )
    )
    with pytest.raises(ValueError):
        await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'connector.db'}"))
    async with open_builtin_contexts(config) as contexts:
        assert len(await (await contexts.get(first.binding.scope_id)).sources.journal.entries()) == 1
        assert (await contexts.connector_checkpoint(first.binding)).checkpoint == first.committed_checkpoint
    del tail["details"]
    path.write_bytes(_archive([root, pending, tail]))
    recovered, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert recovered.previous_checkpoint == first.committed_checkpoint
    assert len(recovered.items) == 2 and len(entries) == 3


@pytest.mark.parametrize("character", ["x", "\u6d77"])
async def test_observation_size_boundary_preserves_full_source_and_preflights_batch(tmp_path, character):
    path = tmp_path / "session.jsonl"
    root = {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "root"}}
    pending = {
        "type": "message",
        "id": "pending",
        "parentId": "root",
        "message": {"role": "user", "content": "pending"},
    }
    tool = {
        "type": "message",
        "id": "tool",
        "parentId": "pending",
        "message": {"role": "toolResult", "toolName": "read", "toolCallId": "call", "content": ""},
    }
    binding = ConnectorBinding(
        scope_id="pi", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    tool["message"]["content"] = character
    event = parse_pi_session(PiSessionInput(content=_archive([root, pending, tool])))[-1]
    observation = project_source_for_transport(registry, await registry.resolve(event))
    bytes_per_character = 3 * len(character.encode("utf-8"))
    overhead = len(observation.model_dump_json().encode()) - bytes_per_character
    tool["padding"] = ""
    overhead += len(',"padding":""')
    count = (MAX_SOURCE_OBSERVATION_BYTES - overhead) // bytes_per_character
    remainder = MAX_SOURCE_OBSERVATION_BYTES - overhead - count * bytes_per_character
    tool["padding"] = "x" * remainder
    full_text = character * count
    tool["message"]["content"] = full_text + character
    path.write_bytes(_archive([root]))
    first, _ = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    path.write_bytes(_archive([root, pending, tool]))
    with pytest.raises(ValueError, match=r"observation.*bytes"):
        await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'connector.db'}"))
    async with open_builtin_contexts(config) as contexts:
        assert len(await (await contexts.get(first.binding.scope_id)).sources.journal.entries()) == 1
        assert (await contexts.connector_checkpoint(first.binding)).checkpoint == first.committed_checkpoint
    tool["message"]["content"] = full_text
    path.write_bytes(_archive([root, pending, tool]))
    completed, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert len(completed.items) == 2 and len(entries) == 3
    stored = entries[-1].source
    assert len(stored.model_dump_json().encode()) == MAX_SOURCE_OBSERVATION_BYTES
    source = PI_EVENT_SOURCE_DEFINITION.source_class.model_validate(stored.payload)
    restored = await registry.read(source)
    assert restored.record == tool
    assert restored.content == f"toolResult: [tool_result read call] {full_text}"
    assert stored.projections[0].value["content"] == restored.content
    replay, entries = await _run_persisted(tmp_path, PiSessionConnector(path), binding)
    assert replay.items == () and len(entries) == 3
