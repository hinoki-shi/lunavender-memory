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

## Prototype result (2026-10-20)

Run with `Ollama qwen3-embedding:latest` (1024 dims), a synthetic corpus of
40 events and 28 queries (including near-duplicate distractors), using
`examples/vector_sidecar_eval.py`:

| method | R@1 | R@5 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| lexical | 0.821 | 0.893 | 0.848 | 0.867 |
| vector | 0.929 | 1.000 | 0.964 | 0.974 |
| equal RRF hybrid | **0.964** | **1.000** | **0.971** | **0.978** |
| vector-weighted RRF | 0.929 | 1.000 | 0.954 | 0.965 |
| coverage-gated hybrid | 0.929 | 1.000 | 0.964 | 0.974 |

The corpus is 40 nodes, synced in 2.18s; 28 query embeddings took 1.68s total
on this machine. This is a plumbing/quality sanity check, not a benchmark.

The important failure case: `和同伴商量出行目的地` had no correct lexical
hit, but lexical produced a one-character spurious match; equal RRF pushed the
correct vector hit from rank 1 to rank 5. A fixed coverage gate fixed that case
but hurt two other queries where a low-coverage lexical hit was correct. The
lesson is to calibrate fusion weights/confidence on a held-out set rather than
hard-coding a gate.

### Conservative fallback

`VectorFallback` was evaluated on the same 28 positive queries plus 6
“should-not-remember” negatives. With its conservative defaults:

- primary: 19
- `use_vector`: 0 without an explicit tag/anchor
- `ask`: 7
- `abstain`: 2
- confident: 19/28, precision **1.000**
- negatives falsely claimed: **0/6**

The fallback is deliberately allowed to say “I am not sure” instead of
injecting a vector-only candidate. It also treats shared anchors as
non-discriminative: if the top two candidates both contain the same query
anchor, the anchor does not lower the acceptance threshold. Thresholds here are
fitted to this synthetic set and the qwen3-embedding score scale; re-tune them
on real data.

### Real-data smoke test (23 nodes)

The existing local runtime DB contained 23 nodes, **0 edges and 0 tags**, from
a single session. It is a cold-start snapshot, not a week of conversation. I
generated 23 queries locally with `qwen2.5:3b` and embedded with
`qwen3-embedding:latest` (no cloud calls):

| method | R@1 | R@5 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| lexical | 0.739 | 0.870 | 0.804 | 0.821 |
| vector | **0.913** | 0.913 | **0.913** | **0.913** |
| equal RRF hybrid | 0.870 | 0.913 | 0.891 | 0.897 |

The fallback returned primary=17, ask=6, confident=17/17, precision **1.000**,
and negative false claims **0/6**. It made **no vector-only auto-use** because
the DB has no tags/entities to serve as exclusive anchors. That is the intended
precision-first behavior: without a verifier or tag extraction, the system
asks instead of asserting a vector-only memory.
