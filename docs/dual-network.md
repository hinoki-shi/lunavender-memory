# Main network and auxiliary network

This repository uses the terms **main network** and **auxiliary network**.
They are not two copies of the same memory.

## Main network

The main network is the graph stored in `nodes`, `tags`, and `edges`:

- episodic and semantic nodes;
- observations frozen at insertion time;
- edges used by `retrieve()`;
- weights, time decay, lifecycle state, and explicit access commits.

The main network is the only source of normal memory retrieval.

## Auxiliary network

The auxiliary network is `SelfTrace` (`memlayer/selftrace.py`), backed by the
`self_trace` table. The normal path is append-only; explicit erasure is
available for deleted memories and user requests. It records:

- retrieval transitions (`retrieval`);
- cognition proposals (`proposal`);
- approvals that create main-network nodes (`approval`);
- cognition-edit bookkeeping (`cognition_edit`);
- reflection-gate decisions (`reflection_gate`).

It does **not** participate in `retrieve()` and is not a second retrieval graph.
Its job is to make changes inspectable: what was proposed, what was rejected,
what parameter/activation set produced a result, and what was explicitly
approved.

## Flow

```text
main graph
   |
   |  retrieve(snapshot, query, d, params)
   v
retrieval result
   |
   |  optional: record_retrieval_if_changed(...)
   v
self_trace: retrieval transition (append-only by default)

working idea
   |
   |  propose_cognition(summary, evidence_ids, ...)
   v
self_trace: proposal (main graph unchanged)
   |
   |  approve_proposal(seq)       <-- explicit main-network vote
   v
reflect(...) -> new semantic node + derived_from edges
   |
   v
self_trace: approval record
```

### 1. Record retrieval only when the conclusion changes

`record_retrieval_if_changed()` compares the current `(produced, rejected)`
signature with the last recorded one. Repeated identical results do not create
new rows. Retrieval records contain ids, a scene fingerprint, `D`, a parameter
fingerprint, and an activation fingerprint rather than the original query or
scene text.

### 2. Propose without writing the main graph

`propose_cognition()` stores a proposal's summary/source-quote fingerprints,
their lengths, and its evidence ids in the auxiliary network. It does not store
the proposal text. It does not create a node or edge. An unapproved proposal
must leave the main graph byte-identical. Because the text is not retained,
`approve_proposal(seq, summary=...)` requires the caller to supply the summary
again at approval time.

### 3. Approve through the normal write path

`approve_proposal(seq, summary=...)` is the main-network vote. It calls `ingest.reflect()`:
the result is a new semantic node plus `derived_from` edges to the evidence.
After that one-time write, the new node follows all normal main-network rules
for retrieval, decay, access, and deletion.

Legacy proposal rows may still contain `summary` / `source_quote` fields. They
are accepted for reading, and the free-text fields are redacted immediately
after approval.

### 4. Keep an explicit edit trail

`record_cognition_edit()` records the reason and optional detail for a future
limited cognition edit. The edit API itself is not implemented yet; the
bookkeeping entry exists so a change cannot become silent later.

### 5. Reflection gate is a proposal filter

`ReflectionGate` is an optional, deterministic policy for deciding whether a
reflection candidate is worth pursuing. Its decision is written to the
auxiliary timeline. It never writes the main graph automatically; an explicit
proposal/approval is still required.

## Invariants

- **Retrieval isolation:** auxiliary rows do not affect `retrieve()` results.
- **Append-only by default:** the normal path only appends and reads.
  `delete_trace()` / `redact_trace_payload()` exist as explicit erasure
  primitives; `forget()` uses them to remove traces that reference deleted
  nodes.
- **Proposal is inert:** proposal does not change nodes or edges.
- **Approval is explicit:** approval uses the normal `reflect()` write path.
- **Deterministic:** fingerprints and narrative rendering use deterministic
  inputs; no model call is required.
- **Bounded narrative:** `narrative()` renders a deterministic text window with
  a character budget and a rough token estimate.

## Minimal example

```python
from memlayer import MemoryStore, Params, SelfTrace

store = MemoryStore(":memory:")
trace = SelfTrace(store)

trace.record_retrieval_if_changed(
    ts=1.0,
    query_terms=("周末", "朋友"),
    d=0.5,
    params=Params(),
    produced_ids=(1, 2),
    rejected_ids=(3,),
    activation_ids=(1, 2, 3),
    scene="某天·下午",
)

seq = trace.propose_cognition(
    ts=2.0,
    summary="用户似乎喜欢周末和朋友去咖啡馆",
    evidence_ids=(1, 2),
    note="explicit evidence",
)

node = trace.approve_proposal(
    seq,
    summary="用户似乎喜欢周末和朋友去咖啡馆",
    created_at=2.0,
)
print(node.id)
print(trace.narrative()["text"])
```

See `tests/test_memory_layer.py::TestSelfTrace` for the invariant tests.

## Current limits

- The auxiliary network is a library primitive, not wired into the example
  runtime by default.
- There is no automatic reflection scheduler or model-backed proposal writer.
- `cognition_edit` currently records an audit entry; the limited edit API and
  drift budget are future work.
- No claim is made that the auxiliary timeline improves retrieval quality by
  itself. Its purpose is inspectability and explicit change control.
