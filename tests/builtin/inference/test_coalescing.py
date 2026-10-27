from __future__ import annotations

import asyncio

import pytest

from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel
from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage


class _Provider:
    profile = EmbeddingProfile(profile_id="boundary", model="offline", dimension=1)

    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(texts)
        return EmbeddingResult(
            vectors=tuple((float(len(text)),) for text in texts),
            usage=InferenceUsage(requests=2, input_tokens=7, output_tokens=0),
        )


async def test_reused_result_objects_keep_distinct_batch_usage_snapshots():
    class ReusingProvider(_Provider):
        def __init__(self):
            super().__init__()
            self.result = EmbeddingResult(vectors=((1.0,),), usage=InferenceUsage(requests=1, input_tokens=0))

        async def embed(self, texts):
            self.calls.append(texts)
            self.result.usage.input_tokens += 7
            return self.result

    provider = ReusingProvider()
    model = CoalescingEmbeddingModel(provider, max_batch_size=1, coalesce_ms=0, cache_size=0)
    first, second = await asyncio.gather(model.embed(("first", "second", "first")), model.embed(("second",)))
    assert provider.calls == [("first",), ("second",)]
    assert first.vectors == ((1.0,), (1.0,), (1.0,))
    assert first.usage == InferenceUsage(requests=2, input_tokens=21)
    assert second.usage == InferenceUsage(requests=1, input_tokens=14)


async def test_inflight_reuse_without_cache_isolated_from_cancelled_waiter():
    class BlockedProvider(_Provider):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts):
            self.started.set()
            await self.release.wait()
            return await super().embed(texts)

    provider = BlockedProvider()
    model = CoalescingEmbeddingModel(provider, max_batch_size=1, coalesce_ms=0, cache_size=0)
    async with asyncio.timeout(5):
        first = asyncio.create_task(model.embed(("shared",)))
        await provider.started.wait()
        late = asyncio.create_task(model.embed(("shared", "shared")))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        provider.release.set()
        result = await late
    assert provider.calls == [("shared",)]
    assert result.vectors == ((6.0,), (6.0,))
    assert result.usage == InferenceUsage(requests=2, input_tokens=7, output_tokens=0)
    await model.embed(("shared",))
    assert provider.calls == [("shared",), ("shared",)]


@pytest.mark.parametrize("failure", [OSError("failed"), asyncio.CancelledError(), None])
async def test_failed_batch_does_not_poison_independent_batch_or_retry(failure):
    class FlakyProvider(_Provider):
        def __init__(self):
            super().__init__()
            self.failed = False

        async def embed(self, texts):
            if texts == ("bad",) and not self.failed:
                self.calls.append(texts)
                self.failed = True
                if failure is not None:
                    raise failure
                return EmbeddingResult(vectors=())
            return await super().embed(texts)

    provider = FlakyProvider()
    model = CoalescingEmbeddingModel(provider, max_batch_size=1, coalesce_ms=0)
    async with asyncio.timeout(5):
        bad, good = await asyncio.gather(model.embed(("bad",)), model.embed(("good",)), return_exceptions=True)
        assert isinstance(bad, type(failure) if failure is not None else ValueError)
        assert good.vectors == ((4.0,),)
        retry = await model.embed(("bad",))
    assert retry.vectors == ((3.0,),)
    assert provider.calls.count(("bad",)) == 2
    assert (await model.embed(("good",))).usage.requests == 0


async def test_query_and_document_embeddings_use_separate_cache_and_batches():
    class AsymmetricProvider(_Provider):
        async def embed_query(self, texts):
            self.calls.append(("query", *texts))
            return EmbeddingResult(vectors=tuple((-float(len(text)),) for text in texts))

    provider = AsymmetricProvider()
    model = CoalescingEmbeddingModel(provider, coalesce_ms=0)
    document, query = await asyncio.gather(model.embed(("same",)), model.embed_query(("same",)))
    assert document.vectors == ((4.0,),)
    assert query.vectors == ((-4.0,),)
    assert (await model.embed(("same",))).vectors == document.vectors
    assert (await model.embed_query(("same",))).vectors == query.vectors
    assert provider.calls == [("same",), ("query", "same")]


@pytest.mark.parametrize("coalesce_ms", [True, -1, float("inf"), float("nan"), 10**400])
def test_invalid_window_rejected_before_scheduling(coalesce_ms):
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(_Provider(), coalesce_ms=coalesce_ms)
