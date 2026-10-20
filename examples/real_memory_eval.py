"""Evaluate lexical/vector/hybrid/fallback on a local memory DB.

Privacy: this script reads the DB path you give it, sends generation requests
only to the local Ollama endpoint, and writes the query set to the output path
(put it outside the repo if it contains personal data). The source DB is only
read; use a copy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent
for p in (str(ROOT), str(EXAMPLES)):
    if p not in sys.path:
        sys.path.insert(0, p)

from memlayer import (  # noqa: E402
    MemoryStore,
    Params,
    Query,
    VectorFallback,
    VectorSidecar,
    content_terms,
    hybrid_seed_scores,
    lexical_relevance,
    retrieve,
    tokenize,
)
from ollama_embedder import OllamaEmbedder  # noqa: E402


def ollama_generate(model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42, "num_predict": 80},
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data.get("response") or "").strip()


def make_query(model: str, summary: str) -> str:
    prompt = (
        "你在为长期记忆检索系统写测试查询。下面是一条记忆摘要：\n"
        + summary
        + "\n请写一句用户可能用来回忆这件事的话。"
        "要求：不要照抄原句；尽量用同义表达替换至少一半的词；"
        "保留人物、地点、物品、时间等关键信息；只输出这一句话，不要解释。"
    )
    text = ollama_generate(model, prompt)
    text = " ".join(text.split()).strip().strip('"“”')
    return text or summary


def rank(ranked: list[int], gold: int) -> dict:
    pos = ranked.index(gold) + 1 if gold in ranked[:10] else None
    return {
        "rank": pos,
        "r1": 1.0 if pos == 1 else 0.0,
        "r5": 1.0 if pos is not None and pos <= 5 else 0.0,
        "mrr": 1.0 / pos if pos else 0.0,
        "ndcg10": 1.0 / math.log2(pos + 1) if pos else 0.0,
    }


def avg(rows: list[dict], k: str) -> float:
    return sum(r[k] for r in rows) / len(rows) if rows else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "real_memory_eval.json"))
    ap.add_argument("--query-model", default="qwen2.5:3b")
    ap.add_argument("--embed-model", default="qwen3-embedding:latest")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--now", type=float, default=None,
                    help="天为单位的时间戳；默认使用当前时间")
    args = ap.parse_args()
    now = args.now if args.now is not None else time.time() / 86400.0

    store = MemoryStore(args.db)
    sidecar = VectorSidecar(":memory:")
    try:
        nodes = [n for n in store.nodes() if len(n.summary.strip()) >= 4][:max(1, args.limit)]
        if not nodes:
            print("FAIL: no usable nodes in DB")
            return 1
        embedder = OllamaEmbedder(model=args.embed_model)
        t0 = time.time()
        sync = sidecar.sync(store, embedder)
        print("nodes:", len(nodes), "sync:", sync,
              "model:", embedder.model, "dim:", embedder.dim,
              "time: %.2fs" % (time.time() - t0))

        queries = []
        for n in nodes:
            q = make_query(args.query_model, n.summary)
            terms = content_terms(q)
            node_terms = set(n.tags) | set(tokenize(n.summary))
            coverage = (len(set(terms) & node_terms) / len(set(terms))
                        if terms else 0.0)
            queries.append({
                "query": q,
                "gold": n.id,
                "coverage": round(coverage, 3),
                "kind": "paraphrase" if coverage < 0.5 else "near_exact",
            })

        snapshot = store.snapshot()
        methods = ["lexical", "vector", "hybrid"]
        rows: dict[str, list[dict]] = {m: [] for m in methods}
        per_query: list[dict[str, object]] = []
        for item in queries:
            q = str(item["query"])
            gold = int(str(item["gold"]))
            terms = content_terms(q)
            lexical = lexical_relevance(snapshot, terms, None)
            vector_hits = sidecar.search(embedder.embed(q), k=10)
            hybrid = hybrid_seed_scores(lexical, vector_hits, limit=10)
            lex_ids = [nid for nid, _ in sorted(
                lexical.items(), key=lambda kv: (-kv[1], kv[0]))]
            vec_ids = [nid for nid, _ in vector_hits]
            hyb_ids = [nid for nid, _ in sorted(
                hybrid.items(), key=lambda kv: (-kv[1], kv[0]))]
            ranks = {"lexical": lex_ids, "vector": vec_ids, "hybrid": hyb_ids}
            detail = {**item, "lexical": None, "vector": None, "hybrid": None}
            for m in methods:
                metric = rank(ranks[m], gold)
                rows[m].append(metric)
                detail[m] = metric["rank"]
            per_query.append(detail)

        print("\nmethod     R@1    R@5    MRR    nDCG@10")
        for m in methods:
            print("%-9s %.3f  %.3f  %.3f  %.3f" % (
                m, avg(rows[m], "r1"), avg(rows[m], "r5"),
                avg(rows[m], "mrr"), avg(rows[m], "ndcg10")))
        print("\nby kind:")
        for kind in ("paraphrase", "near_exact"):
            sub = [d for d in per_query if d["kind"] == kind]
            if not sub:
                continue
            print("  %-10s n=%2d  lex=%s vec=%s hyb=%s"
                  % (kind, len(sub),
                     sum(1 for d in sub if d["lexical"] == 1),
                     sum(1 for d in sub if d["vector"] == 1),
                     sum(1 for d in sub if d["hybrid"] == 1)))

        fallback = VectorFallback(sidecar, embedder)
        fb: list[dict[str, object]] = []
        for item in queries:
            q = str(item["query"])
            gold = int(str(item["gold"]))
            primary = retrieve(snapshot, Query(terms=content_terms(q), now=now),
                               0.5, Params())
            decision = fallback.decide(q, primary, snapshot)
            selected = None
            if decision.decision == "primary" and primary.seeds:
                selected = primary.seeds[0]
            elif decision.decision == "use_vector" and decision.candidates:
                selected = decision.candidates[0]
            fb.append({"decision": decision.decision,
                       "correct": (selected == gold) if selected is not None else None})
        counts: dict[str, int] = {}
        for d in fb:
            dec_key = str(d["decision"])
            counts[dec_key] = counts.get(dec_key, 0) + 1
        for i, f in enumerate(fb):
            per_query[i]["fallback_decision"] = f["decision"]
            per_query[i]["fallback_correct"] = f["correct"]
        confident = [d for d in fb if d["decision"] in ("primary", "use_vector")]
        correct = sum(1 for d in confident if d["correct"])
        print("\nfallback:", counts,
              "confident=%d correct=%d precision=%.3f"
              % (len(confident), correct,
                 correct / len(confident) if confident else 0.0))

        out = {
            "db": args.db,
            "nodes": len(nodes),
            "query_model": args.query_model,
            "embed_model": embedder.model,
            "now_day": round(now, 6),
            "queries": len(queries),
            "metrics": {m: {k: round(avg(rows[m], k), 4)
                            for k in ("r1", "r5", "mrr", "ndcg10")}
                        for m in methods},
            "fallback": {"decisions": counts,
                         "confident": len(confident),
                         "correct": correct,
                         "precision": round(correct / len(confident), 4)
                         if confident else 0.0},
            # Contains personal query text; keep the output outside the repo.
            "rows": per_query,
        }
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("saved (private):", outp)
        return 0
    finally:
        sidecar.close()
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
