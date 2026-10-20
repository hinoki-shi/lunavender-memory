"""Minimal vector-sidecar demo using the deterministic HashEmbedder."""

from __future__ import annotations

import sys
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
    hybrid_seed_scores,
    lexical_relevance,
    remember,
    retrieve,
)


def main() -> None:
    store = MemoryStore(":memory:")
    sidecar = VectorSidecar(":memory:")
    try:
        remember(store, summary="在咖啡馆和朋友聊周末计划",
                 obs=Observation(happy=0.6), scene="周末",
                 tags=("咖啡馆", "朋友"))
        remember(store, summary="去游泳馆练习换气",
                 obs=Observation(happy=0.3), scene="周末",
                 tags=("游泳",))
        remember(store, summary="在家看数学书",
                 obs=Observation(anxiety=0.2), scene="工作日",
                 tags=("数学",))

        emb = HashEmbedder(dim=128)
        print("sync:", sidecar.sync(store, emb))
        query = "咖啡馆"
        lexical = lexical_relevance(store.snapshot(), (query,), "周末")
        vector_hits = sidecar.search(emb.embed(query), k=3)
        fused = hybrid_seed_scores(lexical, vector_hits, limit=3)
        print("lexical:", lexical)
        print("vector:", vector_hits)
        print("fused:", fused)

        result = retrieve(store.snapshot(), Query(terms=(), seed_scores=fused),
                          d=0.5, params=Params())
        print("seeds:", result.seeds)
    finally:
        sidecar.close()
        store.close()


if __name__ == "__main__":
    main()
