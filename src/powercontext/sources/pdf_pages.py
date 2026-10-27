from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import re
from collections import Counter
from copy import copy

from pydantic import BaseModel, Field, field_validator
from pypdf import PageObject, PdfReader
from pypdf.errors import PyPdfError
from pypdf.generic import (
    ArrayObject,
    ContentStream,
    DictionaryObject,
    NameObject,
    NullObject,
    StreamObject,
    TextStringObject,
    create_string_object,
)

from powercontext.builtin.artifacts.memory import MemoryCitation, MemoryHit
from powercontext.builtin.runtime.application import ScopedMemoryApplication
from powercontext.builtin.runtime.models import SearchMemoryRequest
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class _PDFError(ValueError):
    pass


class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            detail = "document_id must not be blank"
            raise _PDFError(detail)
        return value


class PDFPageInput(BaseModel):
    document_id: str
    document_sha256: str
    page_number: int = Field(ge=1)
    text: str


def _page_text(page: PageObject) -> str:
    resources = page.get_inherited("/Resources")
    if resources is not None and not isinstance(resources, (DictionaryObject, NullObject)):
        detail = "PDF page resources must be a dictionary"
        raise _PDFError(detail)
    content = page.get("/Contents")
    if content is not None:
        content = content.get_object()
        streams = content if isinstance(content, ArrayObject) else (content,)
        if any(not isinstance(stream.get_object(), (StreamObject, NullObject)) for stream in streams):
            detail = "PDF page contents must contain only streams"
            raise _PDFError(detail)
    return _with_actual_text(page).extract_text() or ""


def _actual_text(value: object, properties: object) -> str | None:
    if isinstance(value, NameObject) and isinstance(properties, DictionaryObject):
        value = properties.get(value)
        if value is not None:
            value = value.get_object()
    actual = value.get("/ActualText") if isinstance(value, DictionaryObject) else None
    if isinstance(actual, bytes):
        actual = create_string_object(actual)
    if actual is not None and not isinstance(actual, str):
        detail = "PDF ActualText must be a text string"
        raise _PDFError(detail)
    return actual


def _replace_text_operands(operands: list[object], operator: bytes, replacement: str) -> list[object]:
    text = TextStringObject(replacement)
    if operator == b"TJ":
        return [ArrayObject([text])]
    if operator == b'"':
        return [*operands[:2], text]
    return [text]


def _with_actual_text(page: PageObject) -> PageObject:
    contents = page.get("/Contents")
    if contents is None or isinstance(contents.get_object(), NullObject):
        return page
    stream = ContentStream(contents, page.pdf, forced_encoding="bytes")
    resources = page.get_inherited("/Resources")
    properties = resources.get("/Properties") if isinstance(resources, DictionaryObject) else None
    if properties is not None:
        properties = properties.get_object()
    spans: list[str | None] = []
    operations = []
    changed = False
    for operands, operator in stream.operations:
        if operator == b"BDC":
            spans.append(_actual_text(operands[1], properties))
        elif operator == b"BMC":
            spans.append(None)
        elif operator == b"EMC" and spans:
            spans.pop()
        elif operator in (b"Tj", b"TJ", b"'", b'"'):
            index = next((index for index, value in enumerate(spans) if value is not None), None)
            if index is not None:
                operands = _replace_text_operands(operands, operator, spans[index] or "")
                spans[index] = ""
                changed = True
        operations.append((operands, operator))
    if not changed:
        return page
    stream.operations = operations
    result = copy(page)
    result[NameObject("/Contents")] = stream
    return result


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    if not value.content.startswith(b"%PDF-"):
        detail = "input is not a PDF document"
        raise _PDFError(detail)
    digest = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted and not reader.decrypt(""):
            detail = "PDF requires a password"
            raise _PDFError(detail)
        return tuple(
            PDFPageInput(
                document_id=value.document_id, document_sha256=digest, page_number=index, text=_page_text(page)
            )
            for index, page in enumerate(reader.pages, start=1)
        )
    except (PyPdfError, KeyError, TypeError, IndexError, NotImplementedError) as error:
        detail = "PDF cannot be fully parsed"
        raise _PDFError(detail) from error


class _PDFPageSource(Source):
    page: PDFPageInput


class _PDFPageAdapter:
    input_class = PDFPageInput
    name = "pdf-page"
    source_class = _PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> _PDFPageSource:
        encoded = json.dumps([value.document_id, value.document_sha256, value.page_number], ensure_ascii=False).encode()
        return _PDFPageSource(
            name="pdf:" + hashlib.sha256(encoded).hexdigest(),
            materialization=SourceMaterialization.CAPTURED,
            page=value.model_copy(deep=True),
        )

    async def read(self, source: _PDFPageSource, /) -> PDFPageInput:
        return source.page.model_copy(deep=True)


class _PDFTextEvidence:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = _PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: _PDFPageSource, /) -> TextEvidence:
        page = source.page
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{page.document_id}\n{page.text}",
            metadata={
                "document_id": page.document_id,
                "document_sha256": page.document_sha256,
                "page_number": page.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(_PDFPageAdapter(), version="1", projections=(_PDFTextEvidence(),))


async def search_pdf_pages(
    memory: ScopedMemoryApplication, question: str, limit: int = 10
) -> tuple[MemoryCitation, ...]:
    if type(limit) is not int or limit <= 0:
        detail = "limit must be positive"
        raise _PDFError(detail)
    tokens = re.findall(r"[a-z]+|\d+", question.lower())
    stopwords = {
        "a",
        "an",
        "and",
        "answer",
        "any",
        "are",
        "as",
        "assuming",
        "at",
        "be",
        "been",
        "being",
        "by",
        "calculate",
        "can",
        "could",
        "define",
        "details",
        "did",
        "directly",
        "do",
        "does",
        "explicitly",
        "for",
        "from",
        "give",
        "has",
        "have",
        "how",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "only",
        "or",
        "perspective",
        "please",
        "question",
        "relying",
        "reported",
        "respond",
        "response",
        "s",
        "should",
        "shown",
        "state",
        "that",
        "the",
        "their",
        "then",
        "these",
        "this",
        "those",
        "to",
        "use",
        "using",
        "was",
        "were",
        "what",
        "which",
        "who",
        "with",
        "within",
        "would",
        "year",
    }
    terms = list(
        dict.fromkeys(token for token in tokens if token not in stopwords and (len(token) > 1 or token.isdigit()))
    )
    if not terms:
        return ()
    search = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=max(40, limit * 3)))
    snapshot = await memory.list()
    if snapshot.memory_ref is None:
        return ()
    identities: dict[str, tuple[str, ...]] = {}
    documents: dict[str, str] = {}
    for record in snapshot.entries:
        page_sources = tuple(
            sorted(source.source_id for source in record.entry.sources if source.source_type == "pdf-page")
        )
        identities[record.entry.entry_id] = page_sources or (record.entry.entry_id,)
        if page_sources:
            documents[record.entry.entry_id] = record.entry.text.partition("\n")[0]
    hits = tuple(
        MemoryHit(
            memory_ref=record.memory_ref,
            entry_id=record.entry.entry_id,
            entry_version_id=record.entry.entry_version_id,
            text=record.entry.text,
            score=0.0,
            matched_by=("fts",),
        )
        for record in snapshot.entries
    )
    priors = {hit.entry_id: 1 / (30 + rank) for rank, hit in enumerate(search.hits, start=1)}
    preferred = _matching_documents(question, set(documents.values()))
    if preferred:
        hits = tuple(hit for hit in hits if documents.get(hit.entry_id) in preferred)
    elif documents:
        hits = tuple(hit for hit in hits if hit.entry_id in documents)
    return await asyncio.to_thread(_rank_pages, hits, terms, priors, identities, limit)


def _matching_documents(question: str, documents: set[str]) -> set[str]:
    query = re.sub(r"[^a-z0-9]", "", question.lower())
    query_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    matches = set()
    for document in documents:
        name = re.split(r"[_\s](?:19|20)\d{2}", document.lower(), maxsplit=1)[0]
        name = re.sub(r"[^a-z0-9]", "", name)
        if (len(name) > 2 and name in query) or (len(name) > 1 and name in query_words):
            matches.add(document)
    return matches


def _words(text: str) -> list[str]:
    words = []
    for word in re.findall(r"[a-z]+|\d+", text.lower()):
        if word.startswith("cashflow"):
            words.extend(("cash", "flow"))
            continue
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        words.append(word)
    return words


def _query_topics(terms: set[str]) -> tuple[tuple[str, ...], ...]:
    topics = []
    if terms & {
        "ratio",
        "liquidity",
        "asset",
        "debt",
        "capital",
        "cash",
        "margin",
        "ebitda",
        "capex",
        "income",
        "profit",
    }:
        topics.extend([
            (
                "consolidated statements of income",
                "consolidated statements of operations",
                "consolidated statements of earnings",
            ),
            ("consolidated balance sheet", "consolidated statements of financial position"),
            ("consolidated statements of cash flows", "consolidated statement of cash flows"),
        ])
    if terms & {"legal", "litigation", "lawsuit", "battle"}:
        topics.append(("legal proceedings", "litigation", "contingencies"))
    if terms & {"geography", "geographic", "geographical", "region", "country"}:
        topics.extend([
            ("geographic", "geographical", "international", "business segments", "segment information"),
            ("item 1 business", "our business", "business overview"),
        ])
    if terms & {"product", "service", "customer", "segment", "category"}:
        topics.append(("business", "products", "customers", "segments", "segment information"))
    if terms & {"revenue", "sale", "margin", "tax", "change"}:
        topics.append((
            "results of operations",
            "consolidated statements of income",
            "consolidated statements of operations",
        ))
    if "tax" in terms:
        topics.append(("income taxes", "effective tax rate", "provision for income taxes"))
    if terms & {"category", "segment"} and terms & {"revenue", "sale", "growth"}:
        topics.append(("business segment data", "segment results", "segment information", "segment performance"))
    if terms & {"cyclicality", "cyclical"}:
        topics.append(("business", "industry", "demand", "cyclical", "cyclicality"))
    return tuple(topics)


def _topic_score(text: str, phrases: tuple[str, ...]) -> float:
    normalized = " " + " ".join(_words(text)) + " "
    headings = [" " + " ".join(_words(line)) + " " for line in text.splitlines() if 1 <= len(_words(line)) <= 14]
    values = []
    for phrase in phrases:
        normalized_phrase = " " + " ".join(_words(phrase)) + " "
        if normalized_phrase in normalized:
            heading = max(
                (len(normalized_phrase) / len(line) for line in headings if normalized_phrase in line),
                default=0.0,
            )
            values.append(1 + 4 * heading)
    return max(values, default=0.0)


def _rank_pages(
    hits: tuple[MemoryHit, ...],
    terms: list[str],
    priors: dict[str, float],
    identities: dict[str, tuple[str, ...]],
    limit: int,
) -> tuple[MemoryCitation, ...]:
    documents = [hit.text.partition("\n")[2] or hit.text for hit in hits]
    counts = [Counter(_words(text)) for text in documents]
    frequencies = Counter(term for count in counts for term in count)
    query = set(_words(" ".join(terms)))
    topics = _query_topics(query)
    weights = {term: math.log1p((len(hits) - frequencies[term] + 0.5) / (frequencies[term] + 0.5)) for term in query}
    average_length = sum(sum(count.values()) for count in counts) / max(len(counts), 1)
    scores = []
    topic_scores: list[list[float]] = [[] for _ in topics]
    for hit, count, text in zip(hits, counts, documents, strict=True):
        length_factor = 1.2 * (0.25 + 0.75 * sum(count.values()) / max(average_length, 1))
        score = sum(weights[term] * count[term] * 2.2 / (count[term] + length_factor) for term in query if count[term])
        words = _words(text)
        passage = max(
            (
                sum(weights[term] for term in set(words[start : start + 80]) & query)
                for start in range(0, len(words), 40)
            ),
            default=0.0,
        )
        if score:
            scores.append((score + passage, priors.get(hit.entry_id, 0.0), hit))
        for values, phrases in zip(topic_scores, topics, strict=True):
            values.append(_topic_score(text, phrases))
    scores.sort(key=lambda item: (-item[0], -item[1], item[2].entry_id, item[2].entry_version_id))
    selected = _select_pages(hits, scores, topic_scores, priors, identities, limit)
    return tuple(
        MemoryCitation(memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id)
        for hit in selected
    )


def _select_pages(
    hits: tuple[MemoryHit, ...],
    scores: list[tuple[float, float, MemoryHit]],
    topic_scores: list[list[float]],
    priors: dict[str, float],
    identities: dict[str, tuple[str, ...]],
    limit: int,
) -> list[MemoryHit]:
    selected: list[MemoryHit] = []
    selected_pages: set[tuple[str, ...]] = set()
    for _, _, hit in scores:
        if identities[hit.entry_id] not in selected_pages and len(selected) < max(1, limit // 3):
            selected.append(hit)
            selected_pages.add(identities[hit.entry_id])
    rankings = [
        sorted(
            range(len(hits)),
            key=lambda index: (-values[index], -priors.get(hits[index].entry_id, 0.0), hits[index].entry_id),
        )
        for values in topic_scores
    ]
    for rank in range(len(hits)):
        for values, ordering in zip(topic_scores, rankings, strict=True):
            index = ordering[rank]
            if values[index] and identities[hits[index].entry_id] not in selected_pages and len(selected) < limit:
                selected.append(hits[index])
                selected_pages.add(identities[hits[index].entry_id])
        if len(selected) == limit:
            break
    for _, _, hit in scores:
        if identities[hit.entry_id] not in selected_pages and len(selected) < limit:
            selected.append(hit)
            selected_pages.add(identities[hit.entry_id])
    return selected
