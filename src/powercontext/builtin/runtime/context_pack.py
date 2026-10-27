from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import cast

_PACK = "powercontext.context-pack.v1"
_DELTA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    pass


def _wire(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encoded(items: tuple[bytes, ...]) -> list[str]:
    if not isinstance(items, tuple) or not all(isinstance(item, bytes) for item in items):
        detail = "items must be a tuple of bytes"
        raise ContextPackError(detail)
    try:
        for item in items:
            item.decode("utf-8")
    except UnicodeDecodeError as error:
        detail = "chunks must be valid UTF-8"
        raise ContextPackError(detail) from error
    return [base64.b64encode(item).decode("ascii") for item in items]


def _items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        detail = "chunks must be an array of Base64 strings"
        raise ContextPackError(detail)
    try:
        items = tuple(base64.b64decode(item, validate=True) for item in cast(list[str], value))
        for item in items:
            item.decode("utf-8")
    except (ValueError, binascii.Error) as error:
        detail = "invalid Base64 or UTF-8 chunk"
        raise ContextPackError(detail) from error
    if _encoded(items) != value:
        detail = "chunks must use standard Base64 encoding"
        raise ContextPackError(detail)
    return items


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            detail = "duplicate JSON object key"
            raise ContextPackError(detail)
        value[key] = item
    return value


def _payload(raw: bytes, schema: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(raw, bytes):
        detail = "wire payload must be bytes"
        raise ContextPackError(detail)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object)
    except (ValueError, RecursionError) as error:
        detail = "invalid UTF-8 JSON payload"
        raise ContextPackError(detail) from error
    if not isinstance(value, dict) or set(value) != fields | {"schema"} or value.get("schema") != schema:
        detail = "invalid schema or fields"
        raise ContextPackError(detail)
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        detail = "invalid digest"
        raise ContextPackError(detail)
    return value


@dataclass(frozen=True, slots=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        payload: dict[str, object] = {"schema": _PACK, "items": _encoded(items)}
        digest = "sha256:" + hashlib.sha256(_wire(payload)).hexdigest()
        payload["digest"] = digest
        count = 0
        while True:
            payload["byte_count"] = count
            size = len(_wire(payload))
            if size == count:
                return cls(items, digest, count)
            count = size

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if type(chunk_bytes) is not int or chunk_bytes <= 0 or not isinstance(content, str):
            detail = "content must be text and chunk_bytes must be positive"
            raise ContextPackError(detail)
        try:
            content.encode("utf-8")
        except UnicodeEncodeError as error:
            detail = "content must be valid UTF-8 text"
            raise ContextPackError(detail) from error
        chunks: list[bytes] = []
        current = bytearray()
        for character in content:
            encoded = character.encode("utf-8")
            if len(encoded) > chunk_bytes:
                detail = "chunk limit cannot contain a UTF-8 character"
                raise ContextPackError(detail)
            if len(current) + len(encoded) > chunk_bytes:
                chunks.append(bytes(current))
                current.clear()
            current.extend(encoded)
        if current:
            chunks.append(bytes(current))
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        return _wire({
            "schema": _PACK,
            "items": _encoded(self.items),
            "digest": self.digest,
            "byte_count": self.byte_count,
        })

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPack:
        value = _payload(raw, _PACK, {"items", "digest", "byte_count"})
        pack = cls.build(_items(value["items"]))
        if (
            type(value["byte_count"]) is not int
            or value["byte_count"] != pack.byte_count
            or value["digest"] != pack.digest
        ):
            detail = "pack digest or byte count does not match content"
            raise ContextPackError(detail)
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _wire({
            "schema": _DELTA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encoded(self.appended),
        })

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPackDelta:
        value = _payload(raw, _DELTA, {"base_digest", "target_digest", "prefix_items", "appended"})
        prefix = value["prefix_items"]
        if type(prefix) is not int or prefix < 0:
            detail = "prefix_items must be a non-negative integer"
            raise ContextPackError(detail)
        return cls(_digest(value["base_digest"]), _digest(value["target_digest"]), prefix, _items(value["appended"]))


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    verified = ContextPack.from_bytes(base.to_bytes())
    delta = ContextPackDelta.from_bytes(delta.to_bytes())
    if verified.digest != delta.base_digest:
        detail = "delta does not match base digest"
        raise ContextPackError(detail)
    if type(delta.prefix_items) is not int or not 0 <= delta.prefix_items <= len(base.items):
        detail = "delta prefix is out of bounds"
        raise ContextPackError(detail)
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        detail = "delta does not match target digest"
        raise ContextPackError(detail)
    return target
