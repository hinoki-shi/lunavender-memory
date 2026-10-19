#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时间、访问、冻结（§14.1–§14.5 的实现）。

**这张表就是"什么时候动什么"，别的模块不许越界**（§14.6）：

| 时机 | 会动 | 不会动 |
| --- | --- | --- |
| 时间推进 | `recency`、`S_now`（**都是现算，不落盘**） | 观测量、重要性、不确定性、边 |
| 检索命中 | `access_count`、`last_access`、边的强化 | 观测量 |
| 事后重新评价 | **新增**节点 + 建边 | 旧节点一个字节都不改 |

🔑 为什么"时间推进"不写盘也是对的：`recency` 由 `created_at/last_access` + `now` 算出来，
`S_now = S₀ × recency^α`（§16.7-①）。于是"活"的两个量本来就是函数值，
调用方给 `now` 就好 —— 这同时也是 §16.5 那条"同快照跑两遍逐位相同"能成立的原因。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import Params
from .store import MemoryStore
from .types import MemoryState, Node
from .weights import keep_ratio, persistence, s_now


def recency_of(
    created_at: float,
    now: float,
    *,
    last_access: float = 0.0,
    s0: float | None = None,
    access_count: int = 0,
    p: Params,
) -> float:
    """最近性 ∈ (0, 1]：从**上一次被碰到**（访问）或创建时刻起算的遗忘曲线。

    命中检索会把 `last_access` 重置到 `now` ⇒ 下一次 `recency` 回到 1（§14.3"重置到 1"）。

    **曲线形状可切**（§20）：

    * `"exponential"`（可切回；现行默认见 `config.Params.decay_curve = "power"`）：
      `0.5^(t/h)` —— **均匀遗忘**：危险率恒定，最终**严格归零**，没有长尾。
    * `"power"`：`1/(1+t/τ)^β` —— **艾宾浩斯式**：**先掉得快、后面拖着长尾**
      （一年后仍有 5% 上下），更贴合"小时候的事很模糊但还在"。

    🆕 **复述时钟的折扣**（§26.1，`reset_discount_enabled`，**默认关**）：
    检索命中会把 `last_access` 推到 `now`，但**拨回幅度**被 `g(S₀) × h(access_count)` 折扣 ——
    小事件（`S₀` 低于 `reset_weight_gate`）**碰了也等于没碰**，反复提及救不了它（这是调用方 早期 的要求）；
    大事能过闸门 ⇒ 复述可维持 ⇒ 与遗忘达成动态平衡。
    关掉开关时，本函数与旧实现**逐位相同**。
    """

    anchor = max(float(created_at), float(last_access))
    if p.reset_discount_enabled and float(last_access) > float(created_at) and s0 is not None:
        gate = max(1e-9, float(p.reset_weight_gate))
        raw_gate = min(1.0, max(0.0, float(s0) / gate))
        power = max(1e-9, float(p.reset_gate_power))
        gain_gate = raw_gate ** power if power != 1.0 else raw_gate    # g(S₀)：权重闸门（可加陡）
        # `halflife_eff` **随权重缩放**（早期 首轮实测修正）：
        # 大事的复述效果几乎不衰减 ⇒ 能与遗忘达成动态平衡；小事的 halflife→0 ⇒ 一次就无效。
        halflife = max(1e-9, float(p.reset_repeat_halflife) * gain_gate)
        diminish = 1.0 / (1.0 + max(0, int(access_count)) / halflife)  # h(n)：逐级递减
        keep = gain_gate * diminish                                  # 拨回的有效比例 ∈ [0,1]
        anchor = float(created_at) + keep * (float(last_access) - float(created_at))
    age = max(0.0, float(now) - anchor)
    if getattr(p, "decay_curve", "exponential") == "power":
        # 幂律遗忘（Wixted & Ebbesen 1991 那条"遗忘的幂律"）
        tau = max(1e-6, p.power_tau)
        beta = max(1e-6, p.power_beta)
        r = (1.0 + age / tau) ** (-beta)
        return max(1e-9, min(1.0, r))
    half = max(1e-6, p.recency_half_life_days)
    r = 0.5 ** (age / half)
    return max(1e-9, min(1.0, r))


@dataclass(frozen=True)
class NodeView:
    """节点在某一时刻的**现算视图**（不落盘）。检索与报告读的是它。"""

    node: Node
    recency: float
    salience_now: float
    persistence: float
    keep_ratio: float
    # 🆕 §3.3/§20.3 生命周期状态（派生，不落盘）—— 决定"要不要主动想起"
    state: "MemoryState" = None  # type: ignore[assignment]

    @property
    def id(self) -> int:
        return self.node.id


def view_of(node: Node, now: float, p: Params) -> NodeView:
    # §27：有效显著度 = 冻结的 `s0` + 活的 `s_use`（复述累积）。
    # 关掉 `use_growth_enabled` 时 `s_use` 恒为 0 ⇒ 与旧实现逐位相同。
    s_eff = float(node.s0) + float(getattr(node, "s_use", 0.0))
    rec = recency_of(
        node.created_at,
        now,
        last_access=node.last_access,
        s0=s_eff,
        access_count=node.access_count,
        p=p,
    )
    return NodeView(
        node=node,
        recency=rec,
        salience_now=s_now(s_eff, rec, p),
        persistence=persistence(node.base_persist, s_eff, p),
        keep_ratio=keep_ratio(s_eff, p),
        state=state_of(node, now, rec, s_now(s_eff, rec, p), p),
    )


def state_of(node: Node, now: float, recency: float, salience_now: float, p: Params) -> "MemoryState":
    """§3.3/§20.3 的**生命周期状态**（派生量，不落盘、不写图）。

    ⚠️ `ACTIVE` 的两个条件**必须同时满足**：`S_now` 够高 **且**（被访问过 或 还年轻）。
    原因见 `config.Params.state_active_s` 的注释：幂律长尾会让远古旧事一直停在 active 档，
    只按权重判 ⇒ "主动关怀"永远在讲同一件小时候的事。
    """

    if salience_now >= p.state_active_s:
        age_days = max(0.0, float(now) - max(float(node.created_at), float(node.last_access)))
        fresh = node.access_count > 0 or age_days <= p.state_active_max_age_days
        if fresh:
            return MemoryState.ACTIVE
    if salience_now >= p.state_cold_s:
        return MemoryState.COLD
    return MemoryState.ARCHIVED


def advance_time(nodes: Iterable[Node], now: float, p: Params) -> list[NodeView]:
    """"时间推进"= 换一个 `now` 重新现算（**不写盘、不改任何存下来的字段**）。"""

    return [view_of(n, now, p) for n in nodes]


def s_now_of(node: Node, now: float, p: Params) -> float:
    return view_of(node, now, p).salience_now


def commit_access(
    store: MemoryStore,
    node_ids: Sequence[int],
    now: float,
    p: Params,
    *,
    strengthen_edges: bool = False,
    strength: float = 0.05,
) -> dict:
    """检索命中后**唯一**允许的落盘动作（§14.3 / §14.5）。

    * 一定做的：`access_count += 1`、`last_access = now`（复述强化，MemoryBank）；
    * **可选**的：Hebbian 边强化（共同激活的节点之间的边 +strength）。
      ⚠️ 默认关 —— 因为一旦开着，§16.5 的"同 D 同线索跑两遍逐位相同"就**必然失败**
      （第一遍自己改了图）。对照实验里要开，验收里必须关（§16.7-⑤）。

    观测量、`s0`、`scene` **一个都不许动**（§14.3 冻结规则）。
    """

    touched = store.touch_access(list(node_ids), now)
    use_growth = 0
    if p.use_growth_enabled:
        # §27：复述**累积**使用强度（调用方 早期 澄清的主机制）。
        # Δ = rate × g(s0)^p × h(n)，其中 h 的 halflife 还乘上 g ⇒ 大事递减慢、小事递减快。
        nodes = {n.id: n for n in store.nodes_by_ids(list(node_ids))}
        deltas: list[tuple[int, float]] = []
        gate = max(1e-9, float(p.reset_weight_gate))
        power = max(1e-9, float(p.reset_gate_power))
        base_halflife = max(1e-9, float(p.reset_repeat_halflife))
        # 上限 = min(绝对上限, 相对上限 × s0)（早期 代码审查：绝对上限对中位 S₀ 偏大）
        cap_ratio = max(0.0, float(p.use_growth_cap_ratio))
        for nid in {int(i) for i in node_ids}:
            node = nodes.get(nid)
            if node is None:
                continue
            cap = max(0.0, min(float(p.use_growth_cap), cap_ratio * float(node.s0)))
            raw = min(1.0, max(0.0, float(node.s0) / gate))     # 闸门读**冻结的** s0
            g = raw ** power if power != 1.0 else raw
            halflife = max(1e-9, base_halflife * g)
            remain = max(0.0, cap - float(getattr(node, "s_use", 0.0)))
            if remain <= 0.0 or g <= 0.0:
                continue
            diminish = 1.0 / (1.0 + max(0, int(node.access_count)) / halflife)
            delta = min(remain, float(p.use_growth_rate) * g * diminish)
            if delta > 0.0:
                deltas.append((nid, delta))
        use_growth = store.add_s_use(deltas)
    edges_strengthened = 0
    if strengthen_edges:
        edges_strengthened = store.strengthen_edges(list(node_ids), strength)
    return {
        "touched": touched,
        "use_growth": use_growth,
        "edges_strengthened": edges_strengthened,
        "hebbian": bool(strengthen_edges),
    }


def freeze_signature(nodes: Iterable[Node]) -> tuple:
    """冻结量的指纹：观测量 + 场景锚 + 出处。测试用它证明"运行期间一个字节都没变"。

    ⚠️ 这里**故意不含** `access_count`/`last_access` —— 那两个是活的（§14.3）。
    """

    sig = []
    for n in sorted(nodes, key=lambda x: x.id):
        obs = n.obs
        sig.append(
            (
                n.id,
                n.kind.value,
                n.scene,
                tuple(n.tags),
                tuple(sorted(obs.as_dict().items())),
                n.source_quote,
                round(n.s0, 12),
            )
        )
    return tuple(sig)
