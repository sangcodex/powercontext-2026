from __future__ import annotations

import base64
import hashlib
import json

import pytest

from powercontext.builtin.runtime.context_pack import (
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)


@pytest.mark.parametrize("text", ["", "alpha\n", "\u6d77\U0001f600e\u0301\n" * 40])
@pytest.mark.parametrize("chunk_bytes", [4, 5, 512])
def test_unicode_chunking_and_wire_match_independent_reconstruction(text, chunk_bytes):
    pack = ContextPack.from_content(text, chunk_bytes=chunk_bytes)
    assert all(0 < len(chunk) <= chunk_bytes for chunk in pack.items)
    assert "".join(chunk.decode("utf-8") for chunk in pack.items) == text
    assert ContextPack.from_content(text, chunk_bytes=chunk_bytes) == pack
    payload = json.loads(pack.to_bytes())
    unsigned = {"schema": payload["schema"], "items": payload["items"]}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    assert payload["digest"] == "sha256:" + hashlib.sha256(canonical).hexdigest()
    assert pack.byte_count == payload["byte_count"] == len(pack.to_bytes())
    assert b"".join(base64.b64decode(chunk, validate=True) for chunk in payload["items"]) == text.encode()
    assert ContextPack.from_bytes(json.dumps(payload, indent=2).encode()) == pack


@pytest.mark.parametrize("text,limit", [("\ud800", 512), ("good\udfff", 512), ("\U0001f600", 3)])
def test_invalid_unicode_boundaries_raise_contract_error(text, limit):
    with pytest.raises(ContextPackError):
        ContextPack.from_content(text, chunk_bytes=limit)


@pytest.mark.parametrize("raw", [None, "{}", bytearray(b"{}"), b"\xff", b"[" * 2000 + b"0" + b"]" * 2000])
@pytest.mark.parametrize("decoder", [ContextPack.from_bytes, ContextPackDelta.from_bytes])
def test_invalid_wire_has_consistent_error_type(raw, decoder):
    with pytest.raises(ContextPackError):
        decoder(raw)


@pytest.mark.parametrize("kind", ["pack", "delta"])
def test_duplicate_fields_rejected_even_if_last_value_is_valid(kind):
    pack = ContextPack.from_content("base")
    value = pack if kind == "pack" else ContextPackDelta.between(pack, pack)
    raw = b'{"schema":"discarded",' + value.to_bytes()[1:]
    with pytest.raises(ContextPackError):
        type(value).from_bytes(raw)


@pytest.mark.parametrize("prefix", [0, 1, 2])
def test_any_valid_prefix_reconstructs_without_sender_state(prefix):
    base = ContextPack.build((b"first", "\u6d77".encode(), b"old"))
    target = ContextPack.build((b"first", "\u6d77".encode(), b"new", b"tail"))
    wire = json.dumps({
        "schema": "powercontext.context-pack-delta.v1",
        "base_digest": base.digest,
        "target_digest": target.digest,
        "prefix_items": prefix,
        "appended": [base64.b64encode(chunk).decode() for chunk in target.items[prefix:]],
    }).encode()
    result = apply_context_pack_delta(ContextPack.from_bytes(base.to_bytes()), ContextPackDelta.from_bytes(wire))
    assert result == target


@pytest.mark.parametrize("change", [{"byte_count": True}, {"byte_count": 0}, {"digest": "bad"}, {"items": ["YQ==="]}])
def test_tampered_pack_is_rejected(change):
    payload = json.loads(ContextPack.from_content("alpha").to_bytes()) | change
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps(payload).encode())


@pytest.mark.parametrize("appended", [("not-bytes",), (b"\xff",), [b"wrong-container"]])
def test_direct_delta_invalid_chunks_fail_with_contract_error(appended):
    base = ContextPack.build(())
    delta = ContextPackDelta(base.digest, base.digest, 0, appended)
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, delta)
