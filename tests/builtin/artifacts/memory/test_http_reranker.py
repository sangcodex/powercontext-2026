from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryHit
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker


def _candidates(count):
    return tuple(
        MemoryHit(
            memory_ref=ArtifactRef(family="memory", artifact_id="audit", revision=1),
            entry_id=f"entry-{rank}",
            entry_version_id=f"version-{rank}",
            text=f"candidate {rank}",
            score=1.0,
            matched_by=("fts",),
        )
        for rank in range(1, count + 1)
    )


@pytest.mark.parametrize("batch_size", [1, 2, 7])
async def test_integer_precision_and_cross_batch_ties_preserve_global_order(batch_size):
    scores = [2**53, 2**53 + 1, 10**400, -(10**400), 2**53 + 1, 0.5, 0]
    requests = []

    async def response(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "scores": [
                    {"rank": item["rank"], "score": scores[item["rank"] - 1]}
                    for item in reversed(payload["candidates"])
                ]
            },
        )

    candidates = _candidates(len(scores))
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        result = await HTTPMemoryReranker("https://offline.invalid", client=client, batch_size=batch_size).rerank(
            "query", candidates, len(candidates)
        )
        assert not client.is_closed
    assert result.selected_ranks == (3, 2, 5, 1, 6, 7, 4)
    assert result.usage.requests == len(requests) == (len(scores) + batch_size - 1) // batch_size
    assert [item for request in requests for item in request["candidates"]] == [
        {"rank": rank, "text": candidate.text} for rank, candidate in enumerate(candidates, start=1)
    ]


@pytest.mark.parametrize(
    "content",
    [
        b'{"scores":[{"rank":1,"score":0,"score":1}]}',
        b'{"scores":[],"scores":[{"rank":1,"score":1}]}',
        b'{"scores":[{"rank":1,"score":true}]}',
        b'{"scores":[{"rank":1,"score":NaN}]}',
        b'{"scores":[{"rank":1,"score":1e400}]}',
        b'{"scores":[{"rank":1,"score":-Infinity}]}',
        b'{"scores":[{"rank":true,"score":1}]}',
        b'{"scores":[{"rank":1,"score":1,"extra":0}]}',
        b"\xff",
        b"[" * 2000 + b"0" + b"]" * 2000,
    ],
)
async def test_ambiguous_or_invalid_response_fails_with_value_error(content):
    async def response(request):
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(ValueError):
            await HTTPMemoryReranker("https://offline.invalid", client=client).rerank("query", _candidates(1), 1)


async def test_cancelling_http_request_propagates_without_closing_caller_client():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def response(request):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client, asyncio.timeout(5):
        task = asyncio.create_task(
            HTTPMemoryReranker("https://offline.invalid", client=client).rerank("query", _candidates(1), 1)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        assert not client.is_closed
