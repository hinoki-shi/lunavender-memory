# -*- coding: utf-8 -*-
"""向量 sidecar 原型的回归测试（不依赖真实 embedding 模型）。"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memlayer import (  # noqa: E402
    HashEmbedder,
    MemoryStore,
    Observation,
    Params,
    Query,
    VectorSidecar,
    freeze_signature,
    hybrid_seed_scores,
    remember,
    retrieval_text,
    retrieve,
    rrf_fuse,
    text_hash,
)


class BaseCase(unittest.TestCase):
    def mkstore(self) -> MemoryStore:
        store = MemoryStore(":memory:")
        self.addCleanup(store.close)
        return store


class TestVectorSidecar(BaseCase):
    def test_retrieval_text_excludes_source_quote(self) -> None:
        store = self.mkstore()
        node = remember(
            store,
            summary="在咖啡馆和朋友聊周末计划",
            obs=Observation(happy=0.6),
            scene="周末·下午",
            tags=("咖啡馆", "朋友"),
            source_quote="这句话只应该留在主网",
        )
        text = retrieval_text(node)
        self.assertIn("咖啡馆", text)
        self.assertIn("朋友", text)
        self.assertIn("周末·下午", text)
        self.assertNotIn("只应该留在主网", text)
        self.assertEqual(text_hash(text), text_hash(text))

    def test_hash_embedder_is_deterministic_and_normalized(self) -> None:
        emb = HashEmbedder(dim=64)
        a = emb.embed("咖啡馆 周末")
        b = emb.embed("咖啡馆 周末")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        self.assertAlmostEqual(sum(x * x for x in a), 1.0, places=9)
        self.assertNotEqual(a, emb.embed("游泳馆"))

    def test_sync_is_incremental_and_main_graph_is_unchanged(self) -> None:
        store = self.mkstore()
        a = remember(store, summary="在咖啡馆聊周末计划", obs=Observation(happy=0.6),
                     scene="周末", tags=("咖啡馆", "周末"))
        b = remember(store, summary="去游泳馆练习换气", obs=Observation(happy=0.3),
                     scene="周末", tags=("游泳",))
        c = remember(store, summary="在家看数学书", obs=Observation(anxiety=0.2),
                     scene="工作日", tags=("数学",))
        before = freeze_signature(store.nodes())
        emb = HashEmbedder(dim=128)

        with tempfile.TemporaryDirectory() as td:
            sidecar = VectorSidecar(Path(td) / "vectors.sqlite3")
            try:
                first = sidecar.sync(store, emb, now=10.0)
                self.assertEqual(first, {"embedded": 3, "reused": 0, "deleted": 0})
                self.assertEqual(sidecar.stats()["records"], 3)
                second = sidecar.sync(store, emb, now=11.0)
                self.assertEqual(second, {"embedded": 0, "reused": 3, "deleted": 0})
                self.assertEqual(before, freeze_signature(store.nodes()),
                                 "sidecar 同步不许改主网冻结量")

                hits = sidecar.search(emb.embed("咖啡馆"), k=3)
                self.assertTrue(hits)
                self.assertEqual(hits[0][0], a.id)
                filtered = sidecar.search(emb.embed("咖啡馆"), k=3,
                                          allowed_ids={b.id, c.id})
                self.assertTrue(all(nid in {b.id, c.id} for nid, _ in filtered))

                # 修改主网摘要（模拟旧库迁移/测试）→ 只重算这一条
                store.conn.execute("UPDATE nodes SET summary=? WHERE id=?",
                                   ("在咖啡馆聊新的旅行计划", a.id))
                store.conn.commit()
                changed = sidecar.sync(store, emb, now=12.0)
                self.assertEqual(changed["embedded"], 1)

                store.forget([b.id], now=13.0)
                pruned = sidecar.sync(store, emb, now=14.0)
                self.assertEqual(pruned["deleted"], 1)
                self.assertEqual(sidecar.stats()["records"], 2)
            finally:
                sidecar.close()

    def test_model_change_reembeds_all(self) -> None:
        store = self.mkstore()
        remember(store, summary="咖啡馆", obs=Observation())
        remember(store, summary="游泳馆", obs=Observation())
        with tempfile.TemporaryDirectory() as td:
            sidecar = VectorSidecar(Path(td) / "vectors.sqlite3")
            try:
                sidecar.sync(store, HashEmbedder(dim=32, model="hash-v1"))
                out = sidecar.sync(store, HashEmbedder(dim=32, model="hash-v2"))
                self.assertEqual(out["embedded"], 2)
                self.assertEqual(sidecar.stats()["model"], "hash-v2")
            finally:
                sidecar.close()

    def test_bad_embedder_dimension_is_rejected(self) -> None:
        store = self.mkstore()
        remember(store, summary="咖啡馆", obs=Observation())

        class Bad:
            model = "bad"
            dim = 8

            def embed(self, text: str):
                return (1.0, 2.0)

        with tempfile.TemporaryDirectory() as td:
            sidecar = VectorSidecar(Path(td) / "vectors.sqlite3")
            try:
                with self.assertRaises(ValueError):
                    sidecar.sync(store, Bad())
            finally:
                sidecar.close()

    def test_rrf_and_hybrid_seed_scores(self) -> None:
        fused = rrf_fuse([[1, 2, 3], [2, 1]])
        self.assertEqual(set(fused), {1, 2, 3})
        self.assertGreater(fused[1], fused[3])

        hybrid = hybrid_seed_scores(
            {1: 0.9, 2: 0.1},
            [(3, 0.8), (1, 0.7)],
            limit=3,
        )
        self.assertEqual(max(hybrid.values()), 1.0)
        self.assertIn(1, hybrid)
        self.assertIn(3, hybrid)

    def test_retrieve_accepts_external_seed_scores(self) -> None:
        store = self.mkstore()
        a = remember(store, summary="咖啡馆聊周末计划", obs=Observation(),
                     scene="周末", tags=("咖啡馆",))
        b = remember(store, summary="游泳馆练习换气", obs=Observation(),
                     scene="周末", tags=("游泳",))
        q = Query(terms=(), seed_scores={a.id: 0.9, b.id: 0.1})
        result = retrieve(store.snapshot(), q, 0.5, Params())
        self.assertEqual(result.seeds[0], a.id)
        self.assertAlmostEqual(result.seed_scores[a.id], 0.9)

        q2 = Query(terms=(), seed_nodes=(b.id,), seed_scores={b.id: 0.33})
        result2 = retrieve(store.snapshot(), q2, 0.5, Params())
        self.assertEqual(result2.seeds, (b.id,))
        self.assertAlmostEqual(result2.seed_scores[b.id], 0.33)


if __name__ == "__main__":
    unittest.main()
