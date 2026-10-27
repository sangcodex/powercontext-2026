from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NullObject, NumberObject

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources.pdf_pages import PDFDocumentInput, parse_pdf_document, search_pdf_pages


@pytest.mark.parametrize("width_count", [256, 257])
def test_font_width_limit_fails_whole_document_with_value_error(width_count):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    page = writer.add_blank_page(width=100, height=100)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
        NameObject("/FirstChar"): NumberObject(0),
        NameObject("/LastChar"): NumberObject(width_count - 1),
        NameObject("/Widths"): ArrayObject([NumberObject(500)] * width_count),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf (evidence) Tj ET")
    page[NameObject("/Contents")] = stream
    buffer = io.BytesIO()
    writer.write(buffer)
    value = PDFDocumentInput(document_id="font-width-boundary", content=buffer.getvalue())
    if width_count == 257:
        with pytest.raises(ValueError):
            parse_pdf_document(value)
    else:
        pages = parse_pdf_document(value)
        assert [(page.page_number, page.text) for page in pages] == [(1, ""), (2, "evidence")]


@pytest.mark.parametrize(
    "key,value",
    [
        ("/Contents", NumberObject(1)),
        ("/Contents", ArrayObject([NumberObject(1)])),
        ("/Resources", NumberObject(1)),
    ],
)
def test_invalid_page_structure_is_not_silently_treated_as_blank(key, value):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    page = writer.add_blank_page(width=100, height=100)
    page[NameObject(key)] = value
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="invalid-structure", content=buffer.getvalue()))


@pytest.mark.parametrize("contents", [NullObject(), ArrayObject([]), ArrayObject([NullObject()])])
def test_empty_page_content_representations_are_preserved(contents):
    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    page[NameObject("/Contents")] = contents
    buffer = io.BytesIO()
    writer.write(buffer)
    pages = parse_pdf_document(PDFDocumentInput(document_id="blank-page", content=buffer.getvalue()))
    assert [(page.page_number, page.text) for page in pages] == [(1, "")]


@pytest.mark.parametrize("query", ["cedar mineral", "Please describe cedar mineral using the details shown."])
async def test_page_search_recalls_separate_query_aspects_after_reopen(tmp_path, query):
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'search.db'}"))
    documents = [
        "FOREST_REPORT\nCedar roots absorb rainwater.",
        "GEOLOGY_REPORT\nMineral deposits contain quartz crystals.",
        *(f"NOISE_{index}\nPlease describe the details shown using a generic report." for index in range(25)),
    ]
    async with open_builtin_runtime(config) as runtime:
        scope = await runtime.scopes.create(ScopeDraft(title="Pages", summary="Recall", idempotency_key="pages"))
        other = await runtime.scopes.create(ScopeDraft(title="Other", summary="Isolation", idempotency_key="other"))
        memory = runtime.memory.for_scope(scope.scope_id)
        await memory.remember(
            RememberMemoryRequest(entries=tuple(MemoryEntryInput(kind="page", text=text) for text in documents))
        )
        await runtime.memory.for_scope(other.scope_id).remember(
            RememberMemoryRequest(entries=(MemoryEntryInput(kind="page", text="Cedar mineral other scope."),))
        )
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope.scope_id)
        citations = await search_pdf_pages(memory, query, limit=2)
        assert len(citations) == 2
        assert await search_pdf_pages(memory, query, limit=2) == citations
        entries = [(await memory.get(GetMemoryEntryRequest(citation=citation))).entry for citation in citations]
        assert {entry.text for entry in entries} == set(documents[:2])
        assert await search_pdf_pages(memory, "unmatchedneedle", limit=2) == ()
