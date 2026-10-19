#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自我轨迹（self-trace，即设计里的「辅网」）**v1** —— 时间轴版、append-only、只提案不表决。

设计依据：§22（含 22.4 三条硬约束、22.4.1 提案方/表决方、22.9 查新边界）。
本模块 = 逻辑层；**SQL 仍只在 store.py**（"唯一碰 SQL 的文件"那条硬规矩不破）。

三条硬约束的结构化落实：
1. **绝不进主检索**：本模块不参与 `retrieve()` 的任何数据流；用"有/无轨迹行 ⇒
   检索结果逐位相同"钉死（`TestSelfTrace.test_not_in_retrieval`）。
2. **写完冻结**：store 只提供 `append_trace`；本模块没有任何 update/delete 入口。
3. **幂等/可复现**：记录内容全部来自调用方给的确定性输入（逐位签名测试）。

边界（§22.9 查新结论，写论文时照抄）：
* 我们能主张的**只有**「反事实（被拒候选）入账 + 一条不许影响行为的叙事通道」；
* 版本史 / 审计日志 / 时间掩蔽 / 无值摘要 —— **全部是采纳，不是提出**；
* **不许**把它包装成"让她记得更好"（GitOfThoughts 那条禁令）。

范围：
* v0（早期）：轨迹记录 / 提案-批准路 / 成本双轴里的**存储轴** / 三条结构测试；
* v1（早期）：**转变点过滤**（`record_retrieval_if_changed`：结论没变就不再记）＋
  **叙事摘要**（`narrative`：确定性渲染 + 读 token 估算，§22.9.2 的第二条成本轴）。
**未含**：ε 微调接口（现行 spec 的 C-6，另开一轮；前置=累计漂移上限实验）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional, Sequence

from .config import Params
from .ingest import reflect
from .store import MemoryStore
from .types import Modulators, Node, Observation

TRACE_KINDS: tuple[str, ...] = (
    "retrieval", "proposal", "approval", "cognition_edit", "reflection_gate")


def params_fingerprint(params: Params) -> str:
    """参数组指纹（16 hex = 8 字节；§22.3 的"参数指纹(8B)"）。"""

    canon = json.dumps(params.as_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def activation_fingerprint(node_ids: Sequence[int]) -> str:
    """激活集指纹（定序后哈希 ⇒ 与传入顺序无关、可复现）。"""

    canon = ",".join(str(int(i)) for i in sorted({int(i) for i in node_ids}))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


class SelfTrace:
    """时间轴版辅网 v1。

    用法：`st = SelfTrace(store)`；检索后按需 `record_retrieval(...)`（**只记转变点**——
    先比对 `last_retrieval_signature()`）；认知在 `propose_cognition(...)` 里成形，
    经 `approve_proposal(seq)` 才变成主网节点（提案方 / 表决方）。
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    # ---------------- 记录（append-only，全部零 LLM 调用） ----------------
    def record_retrieval(self, *, ts: float, query_terms: Sequence[str], d: float,
                         params: Params, produced_ids: Sequence[int],
                         rejected_ids: Sequence[int] = (),
                         activation_ids: Sequence[int] = (),
                         scene: Optional[str] = None) -> int:
        """记一次检索的"参数与指针"（**不存文本**，§22.3）。

        `rejected_ids` = 反事实：被考虑过但没选中的候选（§22.2-③，**只记 top-2**）。
        """

        payload = {
            "scene": scene,
            "query": " ".join(str(t) for t in query_terms),
            "d": round(float(d), 6),
            "params_fp": params_fingerprint(params),
            "activation_fp": activation_fingerprint(activation_ids),
            "produced": [int(i) for i in produced_ids],
            "rejected": [int(i) for i in rejected_ids][:2],
        }
        return self.store.append_trace(ts=ts, kind="retrieval", payload=payload)

    def record_retrieval_if_changed(self, *, ts: float, query_terms: Sequence[str],
                                    d: float, params: Params,
                                    produced_ids: Sequence[int],
                                    rejected_ids: Sequence[int] = (),
                                    activation_ids: Sequence[int] = (),
                                    scene: Optional[str] = None) -> Optional[int]:
        """**只记转变点**（§22.2-2）：结论没变 ⇒ 返回 None、不落盘。

        判定键 = `(produced, rejected)`（保持顺序：top-k 的次序也是结论的一部分）。
        上一轮与这一轮完全相同 ⇒ 跳过；不同 ⇒ 走 `record_retrieval`。
        """

        sig = (tuple(int(i) for i in produced_ids),
               tuple(int(i) for i in rejected_ids)[:2])
        if self.last_retrieval_signature() == sig:
            return None
        return self.record_retrieval(ts=ts, query_terms=query_terms, d=d, params=params,
                                     produced_ids=produced_ids, rejected_ids=rejected_ids,
                                     activation_ids=activation_ids, scene=scene)

    def last_retrieval_signature(self) -> Optional[tuple]:
        """上一条检索记录的 `(produced, rejected)` —— 调用方比一比，**变了才录**。"""

        for row in reversed(self.store.trace_rows()):
            if row["kind"] == "retrieval":
                p = row["payload"]
                return (tuple(p.get("produced", ())), tuple(p.get("rejected", ())))
        return None

    def propose_cognition(self, *, ts: float, summary: str,
                          evidence_ids: Sequence[int],
                          obs: Optional[Observation] = None,
                          modulators: Optional[Modulators] = None,
                          source_quote: str = "", note: str = "") -> int:
        """辅网**提案**：认知在工作台成形 —— 只进轨迹；**主网一个字节都不动**。

        批准（=把提案变成主网的新认知节点）见 `approve_proposal`。
        """

        payload = {
            "summary": str(summary),
            "evidence": sorted({int(i) for i in evidence_ids}),
            "obs": (obs or Observation()).as_dict(),
            "modulators": asdict(modulators or Modulators()),
            "source_quote": str(source_quote),
            "note": str(note),
        }
        return self.store.append_trace(ts=ts, kind="proposal", payload=payload)

    def approve_proposal(self, seq: int, *, params: Optional[Params] = None,
                         created_at: Optional[float] = None) -> Node:
        """主网**表决**：提案 → 主网新节点（走 `ingest.reflect` 正规写入路径）+ 记一条 approval。

        影响是**一次性写入**，不是持续读取（§22.4.1）—— 生效后它和其他节点一样受主网全部规矩管。
        """

        rows = {r["seq"]: r for r in self.store.trace_rows()}
        prop = rows.get(int(seq))
        if prop is None or prop["kind"] != "proposal":
            raise ValueError(f"没有这个提案：seq={seq}")
        p = prop["payload"]
        evidence = self.store.nodes_by_ids(p.get("evidence", []))
        obs_raw = p.get("obs") or {}
        obs = Observation(**{k: v for k, v in obs_raw.items()
                             if k in Observation.__dataclass_fields__})
        ts = float(created_at if created_at is not None else prop["ts"])
        node = reflect(
            self.store,
            summary=str(p.get("summary", "")),
            sources=evidence,
            obs=obs,
            params=params,
            created_at=ts,
            source_quote=str(p.get("source_quote", "")),
        )
        self.store.append_trace(
            ts=ts, kind="approval",
            payload={"proposal_seq": int(seq), "node_id": int(node.id)})
        return node

    def record_cognition_edit(self, *, ts: float, node_id: int, reason: str,
                              detail: Optional[dict] = None) -> int:
        """认知层微调**记账**（§25.2 铁律 3：每次微调都必须记账）。

        ε 微调接口本身尚未实现（现行 spec 的 C-6）—— 先提供记账入口不成问题：
        没有它，将来"她为什么变了"就不可回答。
        """

        return self.store.append_trace(ts=ts, kind="cognition_edit", payload={
            "node_id": int(node_id), "reason": str(reason), "detail": dict(detail or {})})

    # ---------------- 查询（自我叙事**显式路**；不经过 retrieve） ----------------
    def timeline(self, *, since: Optional[float] = None, until: Optional[float] = None,
                 kinds: Optional[Sequence[str]] = None) -> list[dict]:
        ks = set(kinds) if kinds else None
        out: list[dict] = []
        for row in self.store.trace_rows():
            if since is not None and row["ts"] < float(since):
                continue
            if until is not None and row["ts"] > float(until):
                continue
            if ks is not None and row["kind"] not in ks:
                continue
            out.append(row)
        return out

    # ---------------- 叙事摘要（§22.9.2 的「读 token 轴」取样；只读、零 LLM） ----------------
    def narrative(self, *, since: Optional[float] = None, until: Optional[float] = None,
                  kinds: Optional[Sequence[str]] = None, max_chars: int = 1200) -> dict:
        """把轨迹窗口渲染成**确定性**摘要（同输入同输出；不调模型、不改库）。

        返回：`{"text", "n_records", "n_rendered", "chars", "est_tokens"}`。
        `est_tokens` 是**粗估**（中文按 ~1.5 字/token 折算，零依赖；不是真 tokenizer）。
        """

        rows = self.timeline(since=since, until=until, kinds=kinds)
        lines: list[str] = []
        used_chars = 0
        used = 0
        for row in rows:
            line = _render_trace_line(row)
            if used_chars + len(line) + 1 > max(0, int(max_chars)):
                break
            lines.append(line)
            used_chars += len(line) + 1
            used += 1
        text = "\n".join(lines)
        return {"text": text, "n_records": len(rows), "n_rendered": used,
                "chars": len(text), "est_tokens": int(round(len(text) / 1.5))}

    # ---------------- 成本（双轴里的存储轴；§22.9.2） ----------------
    def cost_report(self) -> dict:
        """粗算存储轴：每条轨迹 vs 每个节点 的平均字节数；读 token 轴见 `narrative()`。"""

        trace = self.store.trace_rows()
        nodes = self.store.nodes()
        trace_bytes = sum(
            len(json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            for r in trace)
        node_bytes = sum(
            len(json.dumps({"summary": n.summary, "obs": n.obs.as_dict(),
                            "tags": list(n.tags), "src": n.source_quote},
                           ensure_ascii=False, sort_keys=True).encode("utf-8"))
            for n in nodes)
        per_t = trace_bytes / len(trace) if trace else 0.0
        per_n = node_bytes / len(nodes) if nodes else 0.0
        return {
            "n_trace": len(trace),
            "n_nodes": len(nodes),
            "bytes_per_trace": round(per_t, 1),
            "bytes_per_node": round(per_n, 1),
            "storage_ratio": round(per_t / per_n, 3) if per_n else None,
        }


def _render_trace_line(row: dict) -> str:
    """单条轨迹 → 一行确定性文本（叙事路用；不展开 retrieval 的原文，因为根本没存）。"""

    ts = datetime.fromtimestamp(row["ts"], tz=timezone.utc).strftime("%m-%d %H:%M UTC")
    p = row["payload"]
    kind = row["kind"]
    if kind == "retrieval":
        prod = ",".join(str(i) for i in p.get("produced", ())) or "—"
        rej = ",".join(str(i) for i in p.get("rejected", ())) or "—"
        return f"- [{ts}] 检索#{row['seq']}：产出[{prod}] 被拒[{rej}]"
    if kind == "proposal":
        ev = ",".join(str(i) for i in p.get("evidence", ())) or "—"
        return f"- [{ts}] 提案#{row['seq']}：{str(p.get('summary', ''))[:40]}（证据[{ev}]）"
    if kind == "approval":
        return (f"- [{ts}] 批准：提案#{p.get('proposal_seq')} → "
                f"主网节点#{p.get('node_id')}")
    if kind == "cognition_edit":
        return (f"- [{ts}] 认知微调#{row['seq']}：节点#{p.get('node_id')} "
                f"因「{str(p.get('reason', ''))[:24]}」")
    if kind == "reflection_gate":
        d = p.get("decision") or {}
        verdict = "放行" if d.get("allowed") else "拦截"
        return (f"- [{ts}] 反思闸门#{row['seq']}：{verdict}"
                f"（{d.get('reason', 'unknown')}）")
    return f"- [{ts}] {kind}#{row['seq']}"
