"""Synthetic quickstart for the core memory package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memlayer import MemoryStore, Observation, Params, Query, remember, retrieve


def main() -> None:
    store = MemoryStore(":memory:")
    try:
        remember(
            store,
            summary="在咖啡馆和朋友聊到周末的计划",
            obs=Observation(happy=0.6, hope=0.4, importance=0.5),
            scene="某天·下午",
            tags=("咖啡馆", "朋友", "周末"),
            source_quote="我们周末去看看吧",
        )
        remember(
            store,
            summary="周末去了书店",
            obs=Observation(happy=0.7, importance=0.4),
            scene="某天·周末",
            tags=("书店", "周末"),
            source_quote="周末去了书店",
        )
        result = retrieve(
            store.snapshot(),
            Query(terms=("周末", "朋友"), now=1.0),
            d=0.5,
            params=Params(),
        )
        print("top:", result.top(k=5))
    finally:
        store.close()


if __name__ == "__main__":
    main()
