#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主动反思触发器 v1 —— 优化价值三闸门（§25.1）。

设计口径：

1. **反事实闸门**：调用方必须先写出一条具体的“当时还可以怎么做”；
2. **复现闸门**：该情境再次发生的概率不低于阈值；
3. **配额**：每日放行上限 + 同一证据集冷却期，防止成本爆炸与反刍。

分工边界（这是本模块最重要的约束）：

* 语义提取（反事实是否具体、复现概率是多少）由上层提取器提供；
* 本模块只做**确定性结构校验与预算控制**，不调模型、不碰主网、不改检索；
* 放行后只返回一个 decision；要形成认知，仍须走
  `SelfTrace.propose_cognition()` → `approve_proposal()` 的提案-批准路。

默认关闭：`ReflectionPolicy(enabled=False)`。关闭时 `evaluate()` 不写轨迹，也不消耗配额。
日界按库内时间单位（天）的整数刻度划分；运行时若用 epoch 秒，须先在 adapter 边界换算成天。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional

from .store import MemoryStore

TRACE_KIND = "reflection_gate"

# 只拦“纯抽象重述”；真正的具体性由调用方写出的 counterfactual 负责。
_ABSTRACT_ONLY = frozenset({
    "她一直在想那件事",
    "他一直很难过，什么都做不了",
    "这事让我很难受",
    "我一直在想为什么会这样",
    "只能这样了",
})


@dataclass(frozen=True)
class ReflectionCandidate:
    """一次候选反思的结构化输入。

    `counterfactual` 应是**具体动作**（例：“下次先问他现在要不要说”），
    不是情绪重述（例：“她一直在想那件事”）。
    `recurrence` ∈ [0,1]，由上层判断“同类情境还会不会再发生”。
    """

    evidence_ids: tuple[int, ...]
    counterfactual: str
    recurrence: float
    scene: str = ""


@dataclass(frozen=True)
class ReflectionPolicy:
    """三闸门的工程初值；默认关。

    ⚠️ `daily_limit=3`、`source_cooldown_days=7`、`min_recurrence=0.5`
    是保守的工程起点，**不是经过真人对齐实验标定的普适阈值**。
    """

    enabled: bool = False
    min_counterfactual_chars: int = 8
    min_recurrence: float = 0.50
    daily_limit: int = 3
    source_cooldown_days: float = 7.0


@dataclass(frozen=True)
class ReflectionDecision:
    """闸门判定结果；`allowed=True` 只代表“可以起一次反思”，不代表已写主网。"""

    allowed: bool
    stage: str
    reason: str
    optimization_value: float
    evidence_key: tuple[int, ...]
    used_today: int
    remaining_today: int
    cooldown_until: Optional[float]

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "stage": self.stage,
            "reason": self.reason,
            "optimization_value": round(float(self.optimization_value), 6),
            "evidence": list(self.evidence_key),
            "used_today": int(self.used_today),
            "remaining_today": int(self.remaining_today),
            "cooldown_until": (round(float(self.cooldown_until), 6)
                               if self.cooldown_until is not None else None),
        }


class ReflectionGate:
    """把三闸门应用到候选上，并把判定记入 self-trace（只增、可复核）。"""

    def __init__(self, store: MemoryStore, policy: Optional[ReflectionPolicy] = None) -> None:
        self.store = store
        self.policy = policy or ReflectionPolicy()

    def evaluate(self, *, ts: float, candidate: ReflectionCandidate,
                 record: bool = True) -> ReflectionDecision:
        """返回判定；开启时默认把判定写入轨迹，供配额与冷却期复核。"""

        p = self.policy
        ts = float(ts)
        evidence = tuple(sorted({int(i) for i in candidate.evidence_ids if int(i) > 0}))
        cf, cf_fp = _normalize_counterfactual(candidate.counterfactual)
        recurrence = float(candidate.recurrence)
        day_index = math.floor(ts)

        allowed_rows = self._allowed_rows()
        used_today = sum(1 for r in allowed_rows if int(r.get("day_index", -1)) == day_index)
        prior_same = [r for r in allowed_rows if tuple(r.get("evidence", ())) == evidence]
        prior_until = (max(float(r["ts"]) for r in prior_same) + p.source_cooldown_days
                       if prior_same else None)
        cooldown_until = prior_until

        stage = "allow"
        reason = "optimization_value"
        allowed = True
        value = 0.0

        if not p.enabled:
            allowed, stage, reason = False, "disabled", "policy_disabled"
        elif not evidence:
            allowed, stage, reason = False, "evidence", "no_evidence"
        elif len(self.store.nodes_by_ids(evidence)) != len(evidence):
            allowed, stage, reason = False, "evidence", "unknown_evidence"
        elif len(cf) < max(1, int(p.min_counterfactual_chars)):
            allowed, stage, reason = False, "counterfactual", "counterfactual_too_short"
        elif cf in _ABSTRACT_ONLY:
            allowed, stage, reason = False, "counterfactual", "counterfactual_abstract"
        elif not math.isfinite(recurrence) or recurrence < 0.0 or recurrence > 1.0:
            allowed, stage, reason = False, "recurrence", "recurrence_out_of_range"
        elif recurrence < float(p.min_recurrence):
            allowed, stage, reason = False, "recurrence", "recurrence_below_threshold"
        elif prior_until is not None and ts < prior_until:
            allowed, stage, reason = False, "cooldown", "source_cooldown"
        elif used_today >= max(0, int(p.daily_limit)):
            allowed, stage, reason = False, "quota", "daily_quota"
        else:
            value = recurrence
            cooldown_until = ts + max(0.0, float(p.source_cooldown_days))

        if allowed:
            used_today += 1
        remaining = max(0, int(p.daily_limit) - used_today)
        decision = ReflectionDecision(
            allowed=allowed,
            stage=stage,
            reason=reason,
            optimization_value=value,
            evidence_key=evidence,
            used_today=used_today,
            remaining_today=remaining,
            cooldown_until=cooldown_until,
        )

        if p.enabled and record:
            self.store.append_trace(ts=ts, kind=TRACE_KIND, payload={
                "day_index": day_index,
                "decision": decision.as_dict(),
                "scene": str(candidate.scene),
                "counterfactual_fp": cf_fp,
                "counterfactual_chars": len(cf),
                "recurrence": round(recurrence, 6),
            })
        return decision

    def decision_rows(self) -> list[dict]:
        """所有已记录的闸门判定（按 seq）；只读。"""

        return [r for r in self.store.trace_rows() if r["kind"] == TRACE_KIND]

    def _allowed_rows(self) -> list[dict]:
        out: list[dict] = []
        for row in self.decision_rows():
            payload = row["payload"]
            decision = payload.get("decision") or {}
            if not decision.get("allowed"):
                continue
            out.append({
                "ts": float(row["ts"]),
                "day_index": int(payload.get("day_index", -1)),
                "evidence": tuple(int(i) for i in decision.get("evidence", ())),
            })
        return out


def _normalize_counterfactual(text: str) -> tuple[str, str]:
    """去掉空白后做长度/抽象校验；轨迹只存摘要指纹，不复制原文。"""

    cf = "".join(str(text).split())
    fp = hashlib.sha256(cf.encode("utf-8")).hexdigest()[:16]
    return cf, fp
