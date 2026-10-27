from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict

from pydantic import BaseModel, Field, field_validator
from pypdf import PageObject, PdfReader
from pypdf.errors import PyPdfError
from pypdf.generic import ArrayObject, DictionaryObject, NullObject, StreamObject

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
    return page.extract_text() or ""


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
    queries = [question, " ".join(terms), *terms]
    scores: dict[tuple[str, int, str, str], float] = defaultdict(float)
    hits: dict[tuple[str, int, str, str], MemoryHit] = {}
    term_ranks: dict[str, dict[tuple[str, int, str, str], int]] = {}
    for query in dict.fromkeys(queries):
        if not query:
            continue
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=min(100, max(40, limit * 3))))
        for rank, hit in enumerate(page.hits, start=1):
            key = (hit.memory_ref.artifact_id, hit.memory_ref.revision, hit.entry_id, hit.entry_version_id)
            scores[key] += 1 / (30 + rank)
            hits[key] = hit
            if query in terms:
                term_ranks.setdefault(query, {})[key] = rank
    if not hits:
        return ()
    ordered = _rank_pages(hits, scores, term_ranks, terms, limit)
    return tuple(
        MemoryCitation(
            memory_ref=hits[key].memory_ref, entry_id=hits[key].entry_id, entry_version_id=hits[key].entry_version_id
        )
        for key in ordered
    )


def _rank_pages(
    hits: dict[tuple[str, int, str, str], MemoryHit],
    scores: dict[tuple[str, int, str, str], float],
    term_ranks: dict[str, dict[tuple[str, int, str, str], int]],
    terms: list[str],
    limit: int,
) -> list[tuple[str, int, str, str]]:
    counts = {key: Counter(re.findall(r"[a-z]+|\d+", hit.text.lower())) for key, hit in hits.items()}
    frequencies = Counter(term for count in counts.values() for term in count)
    average_length = sum(sum(count.values()) for count in counts.values()) / len(counts)
    relevance = {}
    for key, count in counts.items():
        length_factor = 1.2 * (0.25 + 0.75 * sum(count.values()) / max(average_length, 1))
        relevance[key] = sum(
            math.log1p((len(counts) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            * count[term]
            * 2.2
            / (count[term] + length_factor)
            for term in terms
            if count[term]
        )
    ordered: list[tuple[str, int, str, str]] = []
    coverage = dict.fromkeys(terms, 0.0)
    remaining = set(hits)
    maximum = max(relevance.values(), default=1.0) or 1.0
    while remaining and len(ordered) < limit:
        marginal = {
            key: relevance[key] / maximum
            + sum((1 - coverage[term]) / ranks[key] for term, ranks in term_ranks.items() if key in ranks)
            for key in remaining
        }
        chosen = min(remaining, key=lambda key: (-marginal[key], -scores[key], key))
        remaining.remove(chosen)
        ordered.append(chosen)
        for term, ranks in term_ranks.items():
            if chosen in ranks:
                coverage[term] += (1 - coverage[term]) / ranks[chosen]
    return ordered
