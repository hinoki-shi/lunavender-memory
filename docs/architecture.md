# Architecture

The core is intentionally small:

```text
types.py        data contracts: Observation / Node / Edge / GraphSnapshot
config.py       one frozen Params object
store.py        SQLite I/O; the only module that touches SQL
weights.py      salience, persistence, decay helpers
divergence.py   D -> retrieval parameter mapping
lexicon.py      lexical relevance and tokenization
retrieval.py    read-only seed -> bounded diffusion -> ranking
lifecycle.py    time, recency, access commits and state views
ingest.py       remember / reflect / preference induction
privacy.py      sensitive-content gate, hard delete helpers
recent_buffer.py  recent working-memory buffer (pure logic)
selftrace.py    auxiliary self trace (append-only by default, explicit erasure)
reflection_gate.py  optional reflection admission policy
synth.py        synthetic bank for tests and examples
```

The main-network / auxiliary-network split is described in
[dual-network.md](dual-network.md).

## Invariants

- `retrieve()` receives a snapshot and returns a result; it does not mutate the store.
- Observations are frozen after insertion.
- Access changes go through `commit_access()`.
- Cross-scene propagation is explicit; the default path prevents accidental
  co-occurrence shortcuts.
- Privacy deletion is an explicit write, not a retrieval side effect.

## Scope

This repository is a **research preview**. It is useful for studying deterministic
retrieval, forgetting and auditable storage. It is not a complete agent memory
service: extraction, tagging, vector recall and deployment-specific policy are
left to callers.
