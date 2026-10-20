# Vector sidecar

This is an optional, rebuildable vector **index**, not a compressed copy of the
main memory graph.

```text
main graph (SQLite: nodes / edges / tags)
        │  retrieval_text(node) = summary + tags + scene + kind
        ▼
embedding model (real model, or HashEmbedder for tests)
        ▼
node_vectors.sqlite3  (node_id, model, dim, vector, text_hash)
        ▼
cosine search -> vector top-k
        │
lexical top-k ── RRF ──► seed top-k ──► existing bounded diffusion
```

The main graph remains the source of truth. If the sidecar is lost or the
embedding model changes, rebuild it from `MemoryStore.nodes()`. The main graph
is not modified by `sync()` or `search()`.

## Quick start

```python
from memlayer import HashEmbedder, MemoryStore, VectorSidecar

store = MemoryStore("outputs/memory/runtime.sqlite3")
sidecar = VectorSidecar("outputs/memory/vectors.sqlite3")
embedder = HashEmbedder(dim=256)          # test stand-in, not semantic

sidecar.sync(store, embedder)
query_vec = embedder.embed("上次去的咖啡馆")
hits = sidecar.search(query_vec, k=10)
```

`HashEmbedder` is deterministic and dependency-free, but it only matches tokens.
For real semantic recall, implement the `Embedder` protocol around a model such
as BGE-M3 or multilingual-e5:

```python
class MyEmbedder:
    model = "bge-m3"
    dim = 1024

    def embed(self, text: str) -> tuple[float, ...]:
        ...
```

## Incremental sync

`sync()` re-embeds only rows whose `text_hash` changed or whose model/dim
changed. It also deletes rows for nodes no longer present in the main graph:

```python
result = sidecar.sync(store, embedder)
# {"embedded": 3, "reused": 120, "deleted": 1}
```

Call it after `remember()` / `reflect()` (can be asynchronous) and after
`forget()`; `forget()` itself does not know about this optional sidecar.

## Hybrid seeds

Do not add cosine directly to lexical scores. Fuse ranks:

```python
from memlayer import hybrid_seed_scores, lexical_relevance, Query, retrieve

lexical = lexical_relevance(store.snapshot(), ("咖啡馆",), None)
vector_hits = sidecar.search(embedder.embed("咖啡馆"), k=10)
fused = hybrid_seed_scores(lexical, vector_hits, limit=8)

q = Query(terms=(), seed_scores=fused)
result = retrieve(store.snapshot(), q, d=0.5)
```

`hybrid_seed_scores()` uses reciprocal rank fusion and normalizes the result
so it passes the existing `seed_threshold`. The graph diffusion then handles
multi-hop/scene constraints as before.

## Deletion and privacy

- `retrieval_text()` deliberately excludes `source_quote`; the sidecar is not
  a second copy of the raw source text.
- `VectorSidecar.delete(node_ids)` removes vectors explicitly.
- `sync()` prunes rows for nodes that no longer exist.
- The sidecar file is created with `0600` best-effort on POSIX.
- Embeddings are still personal data. Delete them with the memory, and do not
  upload the sidecar as if it were anonymous.

## Scale

`VectorSidecar.search()` is a pure-Python linear scan, intended for the first
prototype and for thousands of nodes. Beyond that, move the same records to
FAISS / HNSW / IVF-PQ. For reproducible experiments, keep a flat inner-product
index; approximate indexes trade recall for latency.

## Evaluation

Build a small query set with four buckets: exact entity, paraphrase, time, and
cross-scene association. Compare lexical-only, vector-only, and hybrid on
Recall@5, MRR, and nDCG@10. A useful hybrid should improve paraphrase recall
without losing exact entity/time queries.
