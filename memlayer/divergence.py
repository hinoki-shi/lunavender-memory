#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发散度 `D`（§16）：**一个旋钮，协调地带动五个参数**。

设计说明：「当前场景相关性这个我们可以用于限制思维发散性，我们改了，
改为发散思维的检索的权重，用于控制发散线索」⇒ 于是 §15 的第五道闸门（逐节点场景门）
被换成了"每次检索一个全局权重"。这正好把 §15.3 担心的"五个参数互相耦合"
从**调参麻烦**变成**有意设计**。

| `D` ↑ | 参数 | 效果 |
| --- | --- | --- |
| ⬆ | `γ`（每跳衰减）上升 | 远处节点也接得住 |
| ⬆ | `D_max`（深度）变大 | 2 跳 → 3 跳 |
| ⬇ | `β`（激活阈值）下降 | 弱激活也保留 |
| ⬆ | `M`（每跳邻居数）变大 | 每个节点铺得更宽 |
| ⬆ | **语义层权重**上升 | 抽象认知更容易浮现（顺带化解 §15-⑤ 的矛盾） |

⚠️ 三条边界（§16.2，本模块**不许**越界）：
① 不改"当时的事实"（D 只影响取回什么，不改任何观测量）；
② 不绕过出处校验；③ 只调召回面，不调真值。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Params
from .types import EdgeKind


@dataclass(frozen=True)
class DParams:
    """`D` 展开后的一次检索参数。**检索只认它，不认 `D` 本身**。"""

    d: float
    beta: float
    depth_max: int
    top_m: int
    semantic_weight: float
    gamma: dict[EdgeKind, float]

    def as_dict(self) -> dict:
        return {
            "D": self.d,
            "beta": self.beta,
            "depth_max": self.depth_max,
            "top_m": self.top_m,
            "semantic_weight": self.semantic_weight,
            "gamma": {k.value: v for k, v in self.gamma.items()},
        }


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def params_for(d: float, p: Params) -> DParams:
    """把一维 `D ∈ [0, 1]` 展开成五个参数（§16.1 的那张表）。"""

    t = clamp01(d)
    gamma_at_full = {
        EdgeKind.CO_OCCUR: p.gamma_co_occur,
        EdgeKind.DERIVED_FROM: p.gamma_derived,
        EdgeKind.PLAY: p.gamma_play,
    }
    gamma = {k: v * lerp(1.0 - p.d_gamma_span, 1.0, t) for k, v in gamma_at_full.items()}
    # 深度：D=0 时收敛到 1 跳（只看直接相关），D=1 时到 depth_max + span
    depth = 1 + int(round(lerp(0.0, float(p.depth_max - 1 + p.d_depth_span), t)))
    return DParams(
        d=t,
        beta=p.beta * lerp(1.0, 1.0 - p.d_beta_span, t),
        depth_max=max(1, depth),
        top_m=max(1, int(round(lerp(float(max(1, p.top_m - p.d_topm_span)), float(p.top_m), t)))),
        semantic_weight=lerp(p.semantic_w_low, p.semantic_w_high, t),
        gamma=gamma,
    )
