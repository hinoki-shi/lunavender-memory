# -*- coding: utf-8 -*-
"""保守向量兜底：主检索失败时只给候选/反问/弃权。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memlayer import (  # noqa: E402
    MemoryStore,
    Observation,
    Params,
    Query,
    VectorFallback,
    VectorSidecar,
    retrieve,
    remember,
)


class FixedEmbedder:
    model = "fixed"
    dim = 2

    def __init__(self, mapping):
        self.mapping = mapping

    def embed(self, text: str) -> tuple[float, ...]:
        return tuple(self.mapping.get(text, (0.0, 1.0)))


class BaseCase(unittest.TestCase):
    def mkstore(self) -> MemoryStore:
        store = MemoryStore(":memory:")
        self.addCleanup(store.close)
        return store

    def mkcase(self, b_vec=(0.5, 0.866)):
        store = self.mkstore()
        a = remember(store, summary="在咖啡馆聊周末旅行", obs=Observation(),
                     scene="周末", tags=("咖啡馆", "旅行"))
        b = remember(store, summary="和同事讨论项目排期", obs=Observation(),
                     scene="工作", tags=("项目",))
        c = remember(store, summary="去游泳馆练习换气", obs=Observation(),
                     scene="周末", tags=("游泳",))
        side = VectorSidecar(":memory:")
        self.addCleanup(side.close)
        side.upsert(a.id, (1.0, 0.0), model="fixed")
        side.upsert(b.id, b_vec, model="fixed")
        side.upsert(c.id, (0.0, 1.0), model="fixed")
        emb = FixedEmbedder({
            "q_accept": (1.0, 0.0),
            "q_ask": (1.0, 0.0),
            "q_low": (-1.0, 0.0),
            "和同伴商量出行目的地": (1.0, 0.0),
        })
        return store, a, b, c, side, emb


class TestVectorFallback(BaseCase):
    def test_primary_confident_does_not_fallback(self):
        store, a, b, c, side, emb = self.mkcase()
        primary = retrieve(store.snapshot(), Query(terms=("咖啡馆",), now=0.0),
                           0.5, Params())
        result = VectorFallback(side, emb).decide("咖啡馆", primary,
                                                   store.snapshot())
        self.assertEqual(result.decision, "primary")
        self.assertEqual(result.candidates, ())

    def test_high_score_and_margin_returns_candidate(self):
        store, a, b, c, side, emb = self.mkcase(b_vec=(0.5, 0.866))
        primary = retrieve(store.snapshot(), Query(terms=("不存在",), now=0.0),
                           0.5, Params())
        result = VectorFallback(side, emb).decide("q_accept", primary,
                                                   store.snapshot())
        self.assertEqual(result.decision, "use_vector")
        self.assertEqual(result.candidates[0], a.id)
        self.assertIn("可能的记忆线索", result.hint)

    def test_close_scores_ask_instead_of_claiming(self):
        store, a, b, c, side, emb = self.mkcase(b_vec=(0.98, 0.199))
        primary = retrieve(store.snapshot(), Query(terms=("不存在",), now=0.0),
                           0.5, Params())
        result = VectorFallback(side, emb).decide("q_ask", primary,
                                                   store.snapshot())
        self.assertEqual(result.decision, "ask")
        self.assertIn("记忆不确定", result.hint)
        self.assertNotEqual(result.decision, "use_vector")

    def test_low_confidence_abstains(self):
        store, a, b, c, side, emb = self.mkcase()
        primary = retrieve(store.snapshot(), Query(terms=("不存在",), now=0.0),
                           0.5, Params())
        result = VectorFallback(side, emb).decide("q_low", primary,
                                                   store.snapshot())
        self.assertEqual(result.decision, "abstain")
        self.assertEqual(result.hint, "")

    def test_spurious_primary_seed_is_ignored(self):
        store, a, b, c, side, emb = self.mkcase(b_vec=(0.2, 0.98))
        # "和同伴..." 与 b 的 "和同事..." 只有单字巧合；主检索会把它当种子。
        terms = tuple("同 伴 商 量 出 行 目 的".split())
        primary = retrieve(store.snapshot(), Query(terms=terms, now=0.0),
                           0.5, Params())
        self.assertTrue(primary.seeds)
        result = VectorFallback(side, emb).decide(
            "和同伴商量出行目的地", primary, store.snapshot())
        self.assertEqual(result.decision, "use_vector")
        self.assertEqual(result.candidates[0], a.id)


if __name__ == "__main__":
    unittest.main()
