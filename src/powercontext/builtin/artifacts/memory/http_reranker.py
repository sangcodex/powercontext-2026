from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference.models import InferenceUsage


class _RerankProtocolError(ValueError):
    pass


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            detail = "response contains a duplicate object key"
            raise _RerankProtocolError(detail)
        result[key] = value
    return result


class HTTPMemoryReranker:
    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            detail = "batch_size must be a positive integer"
            raise _RerankProtocolError(detail)
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(self, query: str, candidates: tuple[MemoryHit, ...], limit: int, /) -> MemoryRerankDecision:
        if not candidates or type(limit) is not int or not 1 <= limit <= len(candidates):
            detail = "candidates and limit are out of bounds"
            raise _RerankProtocolError(detail)
        scores: dict[int, int | float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = [
                {"rank": index + 1, "text": candidates[index].text}
                for index in range(start, min(start + self._batch_size, len(candidates)))
            ]
            response = await self._client.post(self._endpoint, json={"query": query, "candidates": batch})
            requests += 1
            response.raise_for_status()
            try:
                payload = response.json(object_pairs_hook=_json_object)
            except (ValueError, RecursionError) as error:
                detail = "response must be unambiguous valid JSON"
                raise _RerankProtocolError(detail) from error
            if not isinstance(payload, dict) or set(payload) != {"scores"} or not isinstance(payload["scores"], list):
                detail = "response must contain only a scores array"
                raise _RerankProtocolError(detail)
            expected = {item["rank"] for item in batch}
            found: dict[int, int | float] = {}
            for item in payload["scores"]:
                if not isinstance(item, dict) or set(item) != {"rank", "score"}:
                    detail = "score entry has invalid fields"
                    raise _RerankProtocolError(detail)
                rank, score = item["rank"], item["score"]
                if type(rank) is not int or rank not in expected or rank in found:
                    detail = "score ranks do not cover this batch exactly once"
                    raise _RerankProtocolError(detail)
                if type(score) not in {float, int} or (isinstance(score, float) and not math.isfinite(score)):
                    detail = "score must be a finite number"
                    raise _RerankProtocolError(detail)
                found[rank] = score
            if set(found) != expected:
                detail = "response is missing batch scores"
                raise _RerankProtocolError(detail)
            scores.update(found)
        selected = tuple(sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit])
        return MemoryRerankDecision(selected_ranks=selected, usage=InferenceUsage(requests=requests))
