"""Small, non-personal evaluation of lexical vs vector vs hybrid seed recall.

Requires a local Ollama embedding model. Defaults to qwen3-embedding:latest.
The corpus and queries below are synthetic; no user conversation is included.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent
for p in (str(ROOT), str(EXAMPLES)):
    if p not in sys.path:
        sys.path.insert(0, p)

from memlayer import (  # noqa: E402
    MemoryStore,
    Observation,
    Params,
    Query,
    VectorSidecar,
    VectorFallback,
    content_terms,
    hybrid_seed_scores,
    lexical_relevance,
    remember,
    rrf_fuse,
    retrieve,
    tokenize,
)
from ollama_embedder import OllamaEmbedder  # noqa: E402


EVENTS: list[tuple[str, str, tuple[str, ...]]] = [
    ("和小林在咖啡馆讨论暑假去青岛看海", "咖啡馆", ("小林", "青岛", "旅行")),
    ("在咖啡馆把笔记本电脑落在座位上，回去才找到", "咖啡馆", ("电脑", "遗失")),
    ("和同事在咖啡馆争论新项目的排期", "咖啡馆", ("项目", "排期")),
    ("在图书馆复习数学到闭馆", "图书馆", ("数学", "复习")),
    ("在图书馆借了一本关于海洋生物的书", "图书馆", ("海洋", "书")),
    ("在图书馆遇到高中同学小夏", "图书馆", ("小夏", "同学")),
    ("上周三在体育馆打羽毛球扭伤了脚踝", "体育馆", ("羽毛球", "受伤", "上周三")),
    ("体育馆的淋浴间停水，只能提前回家", "体育馆", ("淋浴", "停水")),
    ("在体育馆看了一场篮球决赛", "体育馆", ("篮球", "决赛")),
    ("陪家人去医院做胃镜检查", "医院", ("家人", "胃镜")),
    ("昨天在医院排队时帮一位老人操作自助机", "医院", ("老人", "自助机", "昨天")),
    ("医院门口的药店买到了缺货的药", "医院", ("药店", "药")),
    ("在车站错过了去杭州的高铁", "车站", ("杭州", "高铁")),
    ("在车站帮陌生人找回丢失的行李箱", "车站", ("行李箱", "陌生人")),
    ("车站附近的面馆吃到很好吃的牛肉面", "车站", ("牛肉面", "面馆")),
    ("在家修好了漏水的水龙头", "家里", ("水龙头", "维修")),
    ("在家尝试做戚风蛋糕，结果塌了", "家里", ("蛋糕", "烘焙")),
    ("在家通宵看完了整季纪录片", "家里", ("纪录片", "通宵")),
    ("去年夏天在公园跑步时遇到一只跟着我走的橘猫", "公园", ("橘猫", "跑步", "去年夏天")),
    ("在公园的长椅上读完了一本小说", "公园", ("小说", "长椅")),
    ("公园的湖边有人放风筝", "公园", ("风筝", "湖")),
    ("在公司加班到凌晨改完方案", "公司", ("加班", "方案")),
    ("公司团建去爬山，山顶下雨了", "公司", ("团建", "爬山")),
    ("在公司茶水间听到同事讨论跳槽", "公司", ("跳槽", "茶水间")),
    ("在快餐店吃薯条时看到窗外有只鸟", "快餐店", ("薯条", "鸟")),
    ("在快餐店排队时手机没电了", "快餐店", ("手机", "没电")),
    ("在快餐店尝试了新出的辣味汉堡", "快餐店", ("汉堡", "辣")),
    ("在超市买到了打折的草莓", "超市", ("草莓", "打折")),
    ("在超市把购物卡忘在收银台", "超市", ("购物卡", "收银台")),
    ("在超市遇到邻居阿姨聊了很久", "超市", ("邻居", "聊天")),
    # Near-duplicate distractors: same topic, different entity/time/food.
    ("在图书馆复习物理到闭馆", "图书馆", ("物理", "复习")),
    ("陪朋友去医院做肠镜检查", "医院", ("朋友", "肠镜")),
    ("在车站错过了去上海的高铁", "车站", ("上海", "高铁")),
    ("在公园跑步时遇到一只跟着我走的黑猫", "公园", ("黑猫", "跑步")),
    ("在超市买到了打折的蓝莓", "超市", ("蓝莓", "打折")),
    ("在快餐店吃薯条时看到窗外有只猫", "快餐店", ("薯条", "猫")),
    ("上周四在体育馆打乒乓球扭伤了手腕", "体育馆", ("乒乓球", "受伤", "上周四")),
    ("在咖啡馆把平板电脑落在座位上", "咖啡馆", ("平板", "遗失")),
    ("在家尝试做巧克力蛋糕，结果塌了", "家里", ("巧克力", "蛋糕")),
    ("在公司加班到凌晨改完预算", "公司", ("加班", "预算")),
]

# gold = index into EVENTS
QUERIES: list[tuple[str, int, str]] = [
    ("和同伴商量出行目的地", 0, "paraphrase"),
    ("备考时在阅览室待到很晚", 3, "paraphrase"),
    ("陪亲属去做身体检查", 9, "paraphrase"),
    ("坐火车去外地结果没赶上", 12, "paraphrase"),
    ("在家做甜点结果失败了", 16, "paraphrase"),
    ("在户外活动时被小动物跟着", 18, "paraphrase"),
    ("在办公室熬夜赶完提案", 21, "paraphrase"),
    ("买到了红色的小果子", 27, "paraphrase"),
    ("小林", 0, "exact"),
    ("小夏", 5, "exact"),
    ("胃镜", 9, "exact"),
    ("杭州", 12, "exact"),
    ("水龙头", 15, "exact"),
    ("购物卡", 28, "exact"),
    ("上周三在体育馆打羽毛球", 6, "time"),
    ("昨天在医院排队", 10, "time"),
    ("去年夏天在公园跑步", 18, "time"),
    ("薯条和鸟有什么关系", 24, "association"),
    ("下雨和爬山", 22, "association"),
    ("笔记本电脑落在哪里", 1, "association"),
    ("数学", 3, "exact"),
    ("橘猫", 18, "exact"),
    ("草莓", 27, "exact"),
    ("戚风蛋糕", 16, "exact"),
    ("方案", 21, "exact"),
    ("笔记本电脑", 1, "exact"),
    ("做胃镜之前要注意什么", 9, "association"),
    ("去杭州坐什么车", 12, "association"),
]

NEGATIVE_QUERIES = [
    "我养过一只叫豆豆的狗吗",
    "我上次去巴黎是什么时候",
    "我的吉他放在哪里",
    "公司年会抽中过什么奖",
    "我学过法语吗",
    "我买过一辆自行车吗",
]

STOP = set("的了在是我你他她它这一有不就都也很把被和跟与或吗呢啊吧哦嘛么个些没还又再会让给到过太好多多少什怎什么怎样呀啦对从向当只才已应该能会想说要去看做发生在后前时回上下里外中")


def rank_metric(ranked: list[int], gold: int) -> dict:
    if gold in ranked[:10]:
        rank = ranked.index(gold) + 1
    else:
        rank = None
    return {
        "rank": rank,
        "r1": 1.0 if rank == 1 else 0.0,
        "r5": 1.0 if rank is not None and rank <= 5 else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "ndcg10": (1.0 / math.log2(rank + 1)) if rank else 0.0,
    }


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def normalized_fusion(rankings: list[list[int]]) -> dict[int, float]:
    fused = rrf_fuse(rankings)
    top = max(fused.values(), default=0.0)
    return {nid: s / top for nid, s in fused.items()} if top > 0 else {}


def lexical_coverage(snapshot, terms: tuple[str, ...], node_id: int) -> float:
    """查询内容词被该节点覆盖的比例（用于判断词面命中是不是巧合）。"""

    if not terms:
        return 0.0
    node = snapshot.nodes[node_id]
    node_terms = set(node.tags) | set(tokenize(node.summary))
    return len(set(terms) & node_terms) / len(set(terms))


def main() -> int:
    store = MemoryStore(":memory:")
    sidecar = VectorSidecar(":memory:")
    try:
        nodes = [remember(store, summary=s, obs=Observation(), scene=sc, tags=tags)
                 for s, sc, tags in EVENTS]
        embedder = OllamaEmbedder()
        t0 = time.time()
        sync = sidecar.sync(store, embedder)
        embed_s = time.time() - t0
        if not sidecar.stats()["records"]:
            print("FAIL: sidecar is empty; is Ollama running?")
            return 1
        print("model:", embedder.model, "dim:", embedder.dim,
              "sync:", sync, "time: %.2fs" % embed_s)

        snapshot = store.snapshot()
        methods = ["lexical", "vector", "hybrid", "hybrid_w2", "hybrid_gated"]
        rows: dict[str, list[dict]] = {m: [] for m in methods}
        per_query = []
        q_embed_s = 0.0
        t0 = time.time()
        for text, gold_idx, kind in QUERIES:
            gold = nodes[gold_idx].id
            terms = tuple(t for t in tokenize(text) if t not in STOP)
            lexical = lexical_relevance(snapshot, terms, None)
            vector_hits = sidecar.search(embedder.embed(text), k=10)
            hybrid = hybrid_seed_scores(lexical, vector_hits, limit=10)

            lex_rank = [nid for nid, _ in sorted(
                lexical.items(), key=lambda kv: (-kv[1], kv[0]))]
            vec_rank = [nid for nid, _ in vector_hits]
            hyb_rank = [nid for nid, _ in sorted(
                hybrid.items(), key=lambda kv: (-kv[1], kv[0]))]
            # Vector gets double weight when lexical has a strong spurious hit.
            hyb_w2 = normalized_fusion([vec_rank, vec_rank, lex_rank])
            # If the top lexical hit only matches one incidental character, do
            # not let it drag the seed. Normalized lexical scores can't tell us
            # this (their top is always 1.0), so use raw query coverage.
            top_cov = (lexical_coverage(snapshot, terms, lex_rank[0])
                       if lex_rank else 0.0)
            if top_cov < 0.5:
                gated = normalized_fusion([vec_rank])
            else:
                gated = hybrid
            ranks = {
                "lexical": lex_rank,
                "vector": vec_rank,
                "hybrid": hyb_rank,
                "hybrid_w2": [nid for nid, _ in sorted(
                    hyb_w2.items(), key=lambda kv: (-kv[1], kv[0]))],
                "hybrid_gated": [nid for nid, _ in sorted(
                    gated.items(), key=lambda kv: (-kv[1], kv[0]))],
            }
            detail = {"query": text, "kind": kind,
                      "gold": gold, "gold_summary": EVENTS[gold_idx][0],
                      "lexical_top_coverage": round(top_cov, 3)}
            for m in methods:
                metrics = rank_metric(ranks[m], gold)
                rows[m].append(metrics)
                detail[m] = metrics["rank"]
            per_query.append(detail)
        q_embed_s = time.time() - t0

        print("\nmethod     R@1    R@5    MRR    nDCG@10")
        for m in methods:
            print("%-9s %.3f  %.3f  %.3f  %.3f" % (
                m, mean(rows[m], "r1"), mean(rows[m], "r5"),
                mean(rows[m], "mrr"), mean(rows[m], "ndcg10")))

        print("\nper-query ranks (None = outside top-10)")
        for d in per_query:
            print("- [%s] %s | lex=%s vec=%s hyb=%s w2=%s gate=%s | gold=%s"
                  % (d["kind"], d["query"], d["lexical"], d["vector"],
                     d["hybrid"], d["hybrid_w2"], d["hybrid_gated"],
                     d["gold_summary"]))

        # Conservative fallback: primary first, vector only when primary is
        # missing or has a low-coverage seed; gray zone asks, never claims.
        fallback = VectorFallback(sidecar, embedder)
        fb_pos: list[dict[str, object]] = []
        for text, gold_idx, kind in QUERIES:
            terms = content_terms(text)
            primary = retrieve(snapshot, Query(terms=terms, now=0.0),
                               0.5, Params())
            r = fallback.decide(text, primary, snapshot)
            selected = None
            if r.decision == "primary" and primary.seeds:
                selected = primary.seeds[0]
            elif r.decision == "use_vector" and r.candidates:
                selected = r.candidates[0]
            gold = nodes[gold_idx].id
            fb_pos.append({
                "query": text, "kind": kind, "decision": r.decision,
                "selected": selected, "gold": gold,
                "correct": (selected == gold) if selected is not None else None,
                "reason": r.reason,
            })

        fb_neg: list[dict[str, object]] = []
        for text in NEGATIVE_QUERIES:
            primary = retrieve(snapshot, Query(terms=content_terms(text), now=0.0),
                               0.5, Params())
            r = fallback.decide(text, primary, snapshot)
            fb_neg.append({
                "query": text, "decision": r.decision, "reason": r.reason,
                "false_confident": r.decision in ("primary", "use_vector"),
            })

        decision_count: dict[str, int] = {}
        for d in fb_pos:
            decision = str(d["decision"])
            decision_count[decision] = decision_count.get(decision, 0) + 1
        confident = [d for d in fb_pos if d["decision"] in ("primary", "use_vector")]
        correct = sum(1 for d in confident if d["correct"])
        false_neg = sum(1 for d in fb_neg if d["false_confident"])
        print("\nfallback positive decisions:", decision_count)
        print("fallback positive confident: %d, correct: %d, precision: %.3f"
              % (len(confident), correct,
                 correct / len(confident) if confident else 0.0))
        print("fallback negative: false_confident=%d / %d"
              % (false_neg, len(fb_neg)))

        out = {
            "model": embedder.model,
            "dim": embedder.dim,
            "events": len(EVENTS),
            "queries": len(QUERIES),
            "sync": sync,
            "embed_sync_s": round(embed_s, 3),
            "embed_query_total_s": round(q_embed_s, 3),
            "metrics": {m: {k: round(mean(rows[m], k), 4)
                            for k in ("r1", "r5", "mrr", "ndcg10")}
                        for m in methods},
            "per_query": per_query,
            "fallback": {
                "decisions": decision_count,
                "confident": len(confident),
                "correct": correct,
                "precision": round(correct / len(confident), 4) if confident else 0.0,
                "negative_false_confident": false_neg,
                "negative_total": len(fb_neg),
                "positive_rows": fb_pos,
                "negative_rows": fb_neg,
            },
        }
        outp = ROOT / "outputs" / "vector_sidecar_eval.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("\nsaved:", outp)
        return 0
    finally:
        sidecar.close()
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
