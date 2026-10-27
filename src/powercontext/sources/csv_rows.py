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

"""Capture offline CSV documents as one durable Source per keyed row."""

from __future__ import annotations

import csv
import io
import json
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from urllib.parse import quote

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"
CSV_ROW_ID_PREFIX = "csv:"
_UTF8_BOM = "\ufeff"

NonEmptyText = Annotated[str, Field(min_length=1)]


class CSVDocumentInput(BaseModel):
    """A complete offline CSV document identified by one primary key column."""

    document_id: str
    primary_key: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def require_trimmed_document_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("document_id must be a non-empty string without surrounding whitespace")  # noqa: TRY003
        return value

    @field_validator("primary_key")
    @classmethod
    def require_non_empty_primary_key(cls, value: str) -> str:
        if not value:
            raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003
        return value


class CSVRowInput(BaseModel):
    """One validated CSV record with header order and raw string values preserved."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured row that can be projected back into a citation."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable delimiter-safe identity of one captured CSV row."""

    if not isinstance(document_id, str) or not document_id.strip() or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without surrounding whitespace")  # noqa: TRY003
    if not isinstance(primary_key_value, str):
        raise ValueError("primary_key_value must be a string")  # noqa: TRY003
    identity = f"{CSV_ROW_ID_PREFIX}{quote(document_id, safe='')}"
    if primary_key_value:
        identity = f"{identity}:{quote(primary_key_value, safe='')}"
    else:
        identity = f"{identity}:"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"csv row identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and validate the whole document, then return its accepted rows."""

    text = _decode_document(value.content)
    # Hand the raw text to the reader so quoted line breaks stay inside their field.
    reader = csv.reader(io.StringIO(text, newline=""), dialect="excel", strict=True)
    try:
        header = next(reader, None)
        if header is None:
            raise ValueError("csv document must contain a header record")  # noqa: TRY003
        _validate_header(header, value.primary_key)
        key_index = header.index(value.primary_key)
        rows: list[CSVRowInput] = []
        seen_keys: set[str] = set()
        for record in reader:
            if not record:
                continue
            _validate_record_width(record, len(header))
            primary_key_value = record[key_index]
            _validate_primary_key_value(primary_key_value)
            if primary_key_value in seen_keys:
                raise ValueError(f"duplicate primary key value {primary_key_value!r}")  # noqa: TRY003
            seen_keys.add(primary_key_value)
            rows.append(
                CSVRowInput(
                    document_id=value.document_id,
                    primary_key=value.primary_key,
                    primary_key_value=primary_key_value,
                    columns=tuple(header),
                    values=tuple(record),
                )
            )
    except csv.Error as error:
        raise ValueError(f"malformed csv document: {error}") from error
    return tuple(rows)


def _decode_document(content: bytes) -> str:
    if not isinstance(content, bytes):
        raise TypeError("csv content must be bytes")  # noqa: TRY003
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("csv document must be valid utf-8") from error
    if text.startswith(_UTF8_BOM):
        text = text[len(_UTF8_BOM) :]
    return text


def _validate_header(header: list[str], primary_key: str) -> None:
    if not header:
        raise ValueError("csv header must contain at least one column")  # noqa: TRY003
    seen: set[str] = set()
    for column in header:
        if not column.strip():
            raise ValueError("csv header columns must not be empty or blank")  # noqa: TRY003
        if column in seen:
            raise ValueError(f"duplicate csv header column {column!r}")  # noqa: TRY003
        seen.add(column)
    if primary_key not in seen:
        raise ValueError(f"primary key column {primary_key!r} is not present in the header")  # noqa: TRY003


def _validate_record_width(record: list[str], width: int) -> None:
    if len(record) != width:
        raise ValueError(f"csv record has {len(record)} fields, expected {width}")  # noqa: TRY003


def _validate_primary_key_value(primary_key_value: str) -> None:
    if not primary_key_value.strip():
        raise ValueError("csv primary key value must not be empty or blank")  # noqa: TRY003


class CSVRowSourceAdapter:
    """Resolve rows into captured Sources and read them back losslessly."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            definition_version=CSV_ROW_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one row as ordered, keyed text evidence for retrieval citations."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version=CSV_ROW_SOURCE_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_ID_PREFIX",
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
