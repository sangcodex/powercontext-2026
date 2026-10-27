from __future__ import annotations

import io
import re
import secrets

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NullObject,
    NumberObject,
    TextStringObject,
)

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    RetireMemoryEntryRequest,
    SubmitSourceObservation,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import SourceDefinitionRegistry, manifest_for_definition, project_source_for_transport
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageInput,
    parse_pdf_document,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class _DuplicatePagePipeline:
    async def extract(self, request, /):
        return tuple(
            MemoryEntryInput(
                kind=kind,
                text=TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY)).content + suffix,
                sources=(source,),
            )
            for source in request.sources
            for kind, suffix in (("page", ""), ("duplicate-page", "\nRepeated"))
        )


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


@pytest.mark.parametrize("named_property", [False, True])
@pytest.mark.parametrize("operation", [b"(x) Tj", b"[(x) -12 (y)] TJ", b"(x) '", b'0 0 (x) "'])
def test_actual_text_replaces_nested_spans_once_and_preserves_other_text(named_property, operation):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        NameObject("/Properties"): DictionaryObject({
            NameObject("/Replacement"): DictionaryObject({NameObject("/ActualText"): TextStringObject("ti")})
        }),
    })
    property_value = b"/Replacement" if named_property else b"<< /ActualText (ti) >>"
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf (Sec) Tj /Span "
        + property_value
        + b" BDC /Span << /ActualText (wrong) >> BDC "
        + operation
        + b" EMC (ignored) Tj EMC (on 2022 123.45) Tj "
        + b"/Span << /ActualText () >> BDC (hidden) Tj EMC "
        + b"/Span << /ActualText <FEFF006F00660066006900630065002000630061006600E9> >> "
        + b"BDC (glyphs) Tj EMC ET"
    )
    page[NameObject("/Contents")] = stream
    writer.add_blank_page(width=300, height=300)
    buffer = io.BytesIO()
    writer.write(buffer)
    pages = parse_pdf_document(PDFDocumentInput(document_id="tagged", content=buffer.getvalue()))
    assert re.findall(r"[a-z]+|\d+", pages[0].text.lower()) == ["section", "2022", "123", "45", "office", "caf"]
    assert "café" in pages[0].text
    assert pages[1].text == ""
    assert parse_pdf_document(PDFDocumentInput(document_id="tagged", content=buffer.getvalue())) == pages


def test_invalid_actual_text_fails_whole_document():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    page = writer.add_blank_page(width=100, height=100)
    stream = DecodedStreamObject()
    stream.set_data(b"/Span << /ActualText 123 >> BDC (glyph) Tj EMC")
    page[NameObject("/Contents")] = stream
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="invalid-tag", content=buffer.getvalue()))


def test_nonempty_password_is_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt(user_password=secrets.token_hex(8), owner_password=secrets.token_hex(8))
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="encrypted", content=buffer.getvalue()))


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


@pytest.mark.parametrize("company", ["Orion", "7Q"])
@pytest.mark.parametrize(
    "question", ["Is {company} cashflow conversion improving?", "What is {company}'s quick ratio?"]
)
async def test_statement_pages_survive_lexical_decoys_and_document_mixing(tmp_path, company, question):
    query = question.format(company=company)
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'statements.db'}"))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    values = [
        PDFPageInput(
            document_id="ORION_2030_10K",
            document_sha256="a" * 64,
            page_number=1,
            text="Consolidated Statement of Income\nNet sales 140\nNet income 40",
        ),
        PDFPageInput(
            document_id="ORION_2030_10K",
            document_sha256="a" * 64,
            page_number=2,
            text="Consolidated Balance Sheets\nCurrent assets 80\nCurrent liabilities 30",
        ),
        PDFPageInput(
            document_id="ORION_2030_10K",
            document_sha256="a" * 64,
            page_number=3,
            text="Consolidated Statements of Cash Flows\nOperating activities 90\nCapital expenditures 20",
        ),
        PDFPageInput(
            document_id="OTHER_2030_10K",
            document_sha256="b" * 64,
            page_number=1,
            text="Consolidated Statements of Cash Flows\nCash from operations 999",
        ),
    ]
    values.extend(
        PDFPageInput(
            document_id="ORION_2030_10K",
            document_sha256="a" * 64,
            page_number=index + 4,
            text=f"Business note {index}\nOrion quick ratio cashflow conversion improving outlook. " * 3,
        )
        for index in range(45)
    )
    values = [
        value.model_copy(update={"document_id": value.document_id.replace("ORION", company.upper())})
        for value in values
    ]
    async with open_builtin_runtime(config, candidate_pipeline=_DuplicatePagePipeline()) as runtime:
        scope = await runtime.scopes.create(ScopeDraft(title="Mixed", summary="Statements", idempotency_key="mixed"))
        await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
        source_ids = []
        for value in values:
            source = await registry.resolve(value)
            observation = project_source_for_transport(registry, source)
            await runtime.ingestion.submit(SubmitSourceObservation(scope_id=scope.scope_id, observation=observation))
            source_ids.append(source.name)
        while (await runtime.memory.for_scope(scope.scope_id).flush()).processed:
            pass
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope.scope_id)
        citations = await search_pdf_pages(memory, query, limit=6)
        records = [(await memory.get(GetMemoryEntryRequest(citation=citation))).entry for citation in citations]
        found = [entry.sources[0].source_id for entry in records]
        assert len(found) == len(set(found)) <= 6
        assert set(source_ids[:3]) <= set(found)
        assert source_ids[3] not in found
        retired = next(
            citation
            for citation, entry in zip(citations, records, strict=True)
            if entry.sources[0].source_id == source_ids[2]
        )
        for record in (await memory.list()).entries:
            if record.entry.sources[0].source_id == source_ids[2]:
                latest = next(
                    item for item in (await memory.list()).entries if item.entry.entry_id == record.entry.entry_id
                )
                await memory.retire(RetireMemoryEntryRequest(citation=latest.citation))
        updated = await search_pdf_pages(memory, query, limit=6)
        updated_records = [(await memory.get(GetMemoryEntryRequest(citation=citation))).entry for citation in updated]
        assert source_ids[2] not in {entry.sources[0].source_id for entry in updated_records}
        assert (await memory.get(GetMemoryEntryRequest(citation=retired))).entry.sources[0].source_id == source_ids[2]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
async def test_invalid_search_limits_are_rejected_before_access(limit):
    with pytest.raises(ValueError):
        await search_pdf_pages(None, "evidence", limit=limit)
