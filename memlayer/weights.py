#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""权重公式（§11–§13 的实现，含 §16.7 那轮修订）。

**四个量分四种角色**（§13.1），这里一个都不许混：

| 量 | 角色 | 在哪 |
| --- | --- | --- |
| `S₀` 显著度 | **写入时算一次**（派生，但落盘一份给持久性/压缩用） | 本文件 `salience0()` |
| `S_now` | 检索排序用（含 recency），**不落盘** | `s_now()` / lifecycle |
| 持久性 / 压缩比 | 由 `S₀` 派生 | `persistence()` / `keep_ratio()` |
| 通道向量 | 表达层要"说人话"时的原料（§11.5） | `channel_vector()` |

🔴 **三个反例**（设计文档里反复防的，别在这里犯）：
1. **不许把通道加成总分**（§11.1）：开心 0.8 + 期待 −0.8 相加 = 0，
   会把"情绪复杂"误判成"情绪平淡" ⇒ 显著度用**加权范数**，不是和。
2. **`calm` 与 `importance` 不喂 `S₀`**（§13.5 / §13.2）：前者只在表达层，
   后者只喂压缩阈值。它们进范数就等于"偷偷决定记多牢"。
3. **`uncertainty` 是乘上去的**（§13.4）：不确定的东西不该抢注意力。
"""

from __future__ import annotations

import math

from .config import Params
from .types import CHANNELS, Modulators, Observation


def modulator_multiplier(obs: Observation, mod: Modulators, p: Params) -> dict[str, float]:
    """§14.1 + §14.2 的调制器：**逐通道的乘法倍率**（不是加法）。

    出处：第一次 → 首因/新奇（von Restorff）；当众 → 调用方例子"羞耻几乎只在被看见时成立"。
    """

    lam = p.modulator_lambda
    mult = {c: 1.0 for c in CHANNELS}
    if mod.first_time:
        mult = {c: v * (1.0 + lam) for c, v in mult.items()}
    if mod.public:
        mult["shame"] = mult["shame"] * (1.0 + 2.0 * lam)  # 当众 → 羞耻被显著放大
    if mod.self_relevant:
        mult = {c: v * (1.0 + 0.5 * lam) for c, v in mult.items()}
    if mod.sensory:
        mult = {c: v * (1.0 + 0.25 * lam) for c, v in mult.items()}
    return mult


def channel_vector(obs: Observation, mod: Modulators, p: Params) -> dict[str, float]:
    """**有效通道** = 原始通道 × 调制器（§14.1 的"通道【有效强度】"）。

    表达层要"说人话"（"你那天特别激动"）读的就是它（§11.5：向量必须存下来，
    标量只在排序那一步现算）。
    """

    raw = {c: float(getattr(obs, c)) for c in CHANNELS}
    mult = modulator_multiplier(obs, mod, p)
    return {c: raw[c] * mult[c] for c in CHANNELS}


def salience0(obs: Observation, mod: Modulators, p: Params) -> float:
    """`S₀ = √(Σ wᵢ·(mᵢ·xᵢ)²) × (1 − u) × (1 + κ_u·unfinished)`  —— **加权范数**，不是加权和（§11.2）。

    * 范数的意义：不抵消 ⇒ "既开心又难过"不会变成"情绪平淡"；
      **同理"既盼望又担心"也不会抵消**（hope / worry 是两条独立通道，见 types.CHANNELS）；
    * `(1 − u)`：不确定的不该抢注意力（§14.2）。⚠️ **`u`（uncertainty）已从 v1 撤出**
      （早期 调用方拍板）⇒ 现在恒为 0、这一项恒等于 1；**保留为占位**，
      将来若改成"记忆清晰度"（由压缩比派生）直接启用。
    * `(1 + κ_u·unfinished)`：**未完成的事记得更牢**（Zeigarnik）；
      `unfinished=0` 时这一项 = 1 ⇒ **与加入之前的公式逐位相同**（向后兼容）。
    """

    vec = channel_vector(obs, mod, p)
    weights = p.channel_weights
    acc = 0.0
    for idx, c in enumerate(CHANNELS):
        w = weights[idx] if idx < len(weights) else 1.0
        term = w * vec[c]
        acc += term * term
    base = math.sqrt(acc)
    u = max(0.0, min(1.0, float(obs.uncertainty)))
    unf = max(0.0, min(1.0, float(obs.unfinished)))
    return base * (1.0 - u) * (1.0 + p.kappa_unfinished * unf)


def persistence(base_persist: float, s0: float, p: Params) -> float:
    """`持久性 = base_persist(类型先验) × (1 + κ·S₀)`（§13.3，**这一条是唯一写法**）。

    ⚠️ §12.3 早期写的 `decay_rate = base/(1+κS)` **已作废**（§16.7-③）：
    同一个 `base`/`κ` 在两节里含义互为倒数，会算错。
    """

    return max(1e-6, float(base_persist) * (1.0 + p.kappa * max(0.0, s0)))


def decay_rate(base_persist: float, s0: float, p: Params) -> float:
    """`decay_rate = 1 / 持久性`（§13.3）。"""

    return 1.0 / persistence(base_persist, s0, p)


def keep_ratio(s0: float, p: Params) -> float:
    """`压缩比 = 1 / (1 + μ/S₀)` ∈ (0, 1]：S₀ 越高，丢的细节越少（§12.3）。

    ⚠️ **只作用在 §9.2 的【可丢集合】上**（原话/数字/修饰语）：
    场景锚与维度标签是检索键，**恒留**、不参与压缩（§9.2、§12.0）。
    """

    s = max(0.0, float(s0))
    if s <= 0.0:
        return 0.0
    return 1.0 / (1.0 + p.mu / s)


def s_now(s0: float, recency: float, p: Params) -> float:
    """检索排序用的显著度：`S_now = S₀ × recency^α`（§16.7-① 的拆分）。

    拆分的必要性：`S₀` 的公式里**没有时间项**，所以"它会随时间下降"这件事
    只能由 `S_now` 承担 —— 而 `S₀` 一旦算完就冻结（持久性、压缩阈值读的是它）。
    """

    r = max(0.0, min(1.0, float(recency)))
    return max(0.0, float(s0)) * (r ** p.alpha_recency)


def expressed_arousal(obs: Observation) -> float:
    """**表达层**：`表现出来的激动 = 激动 × (1 − 冷静)`（§13.5）。

    🔑 这一条**不进编码层**：激动与冷静是两根独立的轴，不是一根轴的两端。
    所以它只影响"演出来的样子"，不影响"记多牢"。
    """

    a = max(0.0, min(1.0, float(obs.arousal)))
    c = max(0.0, min(1.0, float(obs.calm)))   # calm 已是派生属性（= 1 − anxiety）
    return a * (1.0 - c)
