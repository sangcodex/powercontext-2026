from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel, embed_query


class _EmbeddingBatchError(ValueError):
    pass


@dataclass(frozen=True, eq=False, slots=True)
class _BatchUsage:
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    def __init__(
        self, model: EmbeddingModel, *, max_batch_size: int = 64, coalesce_ms: float = 5.0, cache_size: int = 1024
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            detail = "max_batch_size must be a positive integer"
            raise _EmbeddingBatchError(detail)
        if type(cache_size) is not int or cache_size < 0:
            detail = "cache_size must be a non-negative integer"
            raise _EmbeddingBatchError(detail)
        if type(coalesce_ms) not in {int, float} or coalesce_ms < 0:
            detail = "coalesce_ms must be finite and non-negative"
            raise _EmbeddingBatchError(detail)
        try:
            delay = coalesce_ms / 1_000
        except OverflowError as error:
            detail = "coalesce_ms is outside the supported range"
            raise _EmbeddingBatchError(detail) from error
        if not math.isfinite(delay):
            detail = "coalesce_ms must be finite and non-negative"
            raise _EmbeddingBatchError(detail)
        self.profile = model.profile
        self._model = model
        self._batch_size = max_batch_size
        self._delay = delay
        self._cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str], EmbeddingVector] = OrderedDict()
        self._pending: dict[tuple[str, str], asyncio.Future[tuple[EmbeddingVector, _BatchUsage]]] = {}
        self._queued: list[tuple[str, str]] = []
        self._scheduled = False
        self._tasks: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        return await self._embed(texts, "document")

    async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        from powercontext.builtin.inference.protocols import QueryEmbeddingModel

        return await self._embed(texts, "query" if isinstance(self._model, QueryEmbeddingModel) else "document")

    async def _embed(self, texts: tuple[str, ...], purpose: str) -> EmbeddingResult:
        vectors: dict[str, EmbeddingVector] = {}
        waiting: dict[str, asyncio.Future[tuple[EmbeddingVector, _BatchUsage]]] = {}
        loop = asyncio.get_running_loop()
        for text in texts:
            key = (purpose, text)
            if key in self._cache:
                self._cache.move_to_end(key)
                vectors[text] = self._cache[key]
            else:
                if key not in self._pending:
                    future = loop.create_future()
                    future.add_done_callback(_consume_exception)
                    self._pending[key] = future
                    self._queued.append(key)
                waiting[text] = self._pending[key]
        if self._queued and not self._scheduled:
            self._scheduled = True
            task = asyncio.create_task(self._dispatch())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        results = await asyncio.gather(*(asyncio.shield(future) for future in waiting.values()))
        batches: set[_BatchUsage] = set()
        for text, (vector, batch) in zip(waiting, results, strict=True):
            vectors[text] = vector
            batches.add(batch)
        usage = InferenceUsage(
            requests=sum(batch.usage.requests for batch in batches),
            input_tokens=_token_sum(tuple(batch.usage.input_tokens for batch in batches)),
            output_tokens=_token_sum(tuple(batch.usage.output_tokens for batch in batches)),
        )
        return EmbeddingResult(vectors=tuple(vectors[text] for text in texts), usage=usage)

    async def _dispatch(self) -> None:
        await asyncio.sleep(self._delay)
        queued, self._queued = self._queued, []
        self._scheduled = False
        for purpose in ("document", "query"):
            keys = [key for key in queued if key[0] == purpose]
            for start in range(0, len(keys), self._batch_size):
                task = asyncio.create_task(self._run_batch(tuple(keys[start : start + self._batch_size]), purpose))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    async def _run_batch(self, keys: tuple[tuple[str, str], ...], purpose: str) -> None:
        texts = tuple(key[1] for key in keys)
        try:
            result = await (embed_query(self._model, texts) if purpose == "query" else self._model.embed(texts))
            _validate_batch(result, len(keys))
            batch = _BatchUsage(result.usage.model_copy(deep=True))
        except (Exception, asyncio.CancelledError) as error:
            for key in keys:
                self._pending.pop(key).set_exception(error)
            return
        for key, vector in zip(keys, result.vectors, strict=True):
            if self._cache_size:
                self._cache[key] = vector
                self._cache.move_to_end(key)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            self._pending.pop(key).set_result((vector, batch))


def _validate_batch(result: EmbeddingResult, expected: int) -> None:
    if len(result.vectors) != expected:
        detail = "provider vector count does not match batch size"
        raise _EmbeddingBatchError(detail)


def _consume_exception(future: asyncio.Future[tuple[EmbeddingVector, _BatchUsage]]) -> None:
    if not future.cancelled():
        future.exception()


def _token_sum(values: tuple[int | None, ...]) -> int | None:
    return sum(value for value in values if value is not None) if any(value is not None for value in values) else None
