#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆层 · **数据契约**（零依赖层）。

依赖方向：`types` ← 所有人；`types` **不 import 任何本项目模块、不碰 SQL**。
这是防耦合的第一条硬规矩：检索不 import 存储，存储不 import 检索，
两者都只 import 这里 ⇒ 将来把存储换成向量库，检索一行都不用改。

对应设计：§3.1（三张表）、§9.3（情景层 vs 语义层）、§14.1（观测量原子）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional


class NodeKind(str, Enum):
    """§9.3：抽象（语义层）**只能新增**，不能替换情景节点。"""

    EPISODE = "episode"    # 情景层：有场景锚（时间 + 地点 + 在场的人 + 话题）
    SEMANTIC = "semantic"  # 语义层：抽象认知，**无场景锚**（§15-⑤ 豁免的那一类）


class MemoryState(str, Enum):
    """§3.3 / §20.3 的生命周期状态 —— **派生量**（由 `S_now` 与访问史算出来，**不落盘**）。

    它决定的是**"要不要主动想起"**，而不是"还在不在"：

    | 状态 | 含义 | 行为 |
    | --- | --- | --- |
    | `ACTIVE` | 还会自己浮上来 | 主动关怀的候选；排序优先 |
    | `COLD` | 不主动，但一提起就想得起来 | 正常检索可取回 |
    | `ARCHIVED` | 只剩**场景锚 + 维度标签 + 摘要**，细节丢了 | "想不起来，但一提就想起来" |

    ⚠️ **归档 ≠ 删除**（§2）：节点一直在、`source_quote` 也一直在。
    早期 的 365 天实验证明"一辈子忘不了"靠的是**不删 + 线索可达**，不是靠权重不掉。
    """

    ACTIVE = "active"
    COLD = "cold"
    ARCHIVED = "archived"


class EdgeKind(str, Enum):
    """§15.1③ 的边类型表（第三种 `play` 是 §17 的，默认禁用）。"""

    CO_OCCUR = "co_occur"          # 共享实体/标签 ⇒ **只能同场景内传播**
    DERIVED_FROM = "derived_from"  # 反思派生 ⇒ **唯一的跨场景通道**
    PLAY = "play"                  # §17 玩梗派生 ⇒ 默认关，且只允许高 D 走到


# 喂显著度 S₀ 的**情绪通道**。
# ⚠️ 这里**没有** calm、importance（§16.7-②：calm 只在表达层，importance 只喂压缩阈值）；
#    uncertainty 也不是通道 —— 它是以 (1−u) **乘**在 S₀ 上的（§14.2）。
#
# 🔴 **`hope` 与 `worry` 是两条独立通道**（早期 设计说明，与 §11.1 同一条道理）：
#    "既盼着考完解放、又担心考砸"是**同时成立**的。若把它们合成一根 −4~+4 的轴（相减），
#    这种"既盼又怕"会被抵消成 0 —— 也就是把"情绪复杂"误判成"情绪平淡"，正是 §11.1 那个致命反例。
#    ⇒ 两条各自进范数（范数不抵消），`anticipation = hope − worry` **只留作表达层/兼容**。
# 🔴 早期口径：`calm` **从"表达层调制器"改回一条普通通道**。
#    实际判据是「**内心里有没有持续的不安/张力**」——纯等待（等消息、反复查看进度）
#    本身是**平静**的；有具体的"没底/慌"才是低。这跟 §13.5 原来的"表达抑制"语义不同，
#    而且**反复做一件事不算情绪波动**（不能把"行为反复"直接读成 agitation）。
# 🔴 第二次改：`calm` → **`anxiety`（不安）**。
#    原因：人工标注实际是按"**不安度**"打的（0=内心很平静、4=一直悬着/发慌），
#    而早期页面把方向写反了。
#    ⇒ 内部统一按"不安"存；`calm = 1 − anxiety` 降级为**派生值**（只为兼容旧代码/表达层）。
CHANNELS: tuple[str, ...] = ("happy", "angry", "shame", "hope", "worry", "anxiety", "arousal")


@dataclass(frozen=True)
class Observation:
    """一个事件的**观测量·原子**（§14.1）：提取器给一次，之后**永久冻结**。

    冻结的理由（§14.3）：否则会"越提越生气" —— 复述把情绪推高 ⇒ 人设漂移。
    要改只能**新增节点**（"我现在回头看那次，觉得没那么严重"）+ 建边指回来。
    """

    happy: float = 0.0
    angry: float = 0.0
    shame: float = 0.0
    anticipation: float = 0.0   # **派生量** = hope − worry：只用于表达/兼容，**不再进 S₀**
    hope: float = 0.0           # 盼望好事（0~1）——**独立通道**
    worry: float = 0.0          # 担心坏事（0~1）——**独立通道**
    arousal: float = 0.0        # 激动：**唯一进编码层的强度轴**（§13.5）
    anxiety: float = 0.0        # **不安**（0=内心很平静，1=一直悬着/发慌）—— 进 S₀ 的通道
    importance: float = 0.0     # 重要性：只喂压缩阈值，**不喂 S**（§13.2）
    uncertainty: float = 0.0    # 不确定性：以 (1−u) 乘在 S₀ 上（§13.4）
    unfinished: float = 0.0     # **未完成/还会反复想起**（0~1）：`(1 + κ_u·unfinished)` 乘在 S₀ 上。
    #   设计缺口：等待期的激动可能不高，但这类事往往记得很久 ⇒
    #   「记得牢」不只由当下激不激动决定，还有**事情没完结、反复想起**（Zeigarnik 效应）。

    def as_dict(self) -> dict[str, float]:
        return {
            "happy": self.happy,
            "angry": self.angry,
            "shame": self.shame,
            "anticipation": self.anticipation,
            "hope": self.hope,
            "worry": self.worry,
            "arousal": self.arousal,
            "anxiety": self.anxiety,
            "importance": self.importance,
            "uncertainty": self.uncertainty,
            "unfinished": self.unfinished,
        }

    @property
    def calm(self) -> float:
        """**派生值** = `1 − anxiety`（旧代码/表达层还在读它；§13.5 的"表达抑制"那套已搁置）。"""

        return 1.0 - self.anxiety


@dataclass(frozen=True)
class Modulators:
    """§14.1 的调制器：**乘法**作用在通道上（不是加到通道里）。

    §11.3：调用方那句"第一次 × 快乐 = 倍乘"要抽象成**一般机制**，不是一条特例。
    """

    first_time: bool = False     # 第一次（首因/新奇，von Restorff）
    public: bool = False         # 当众（调用方例子：羞耻几乎只在被看见时成立）
    self_relevant: bool = False  # 自我相关
    sensory: bool = False        # 感官细节


@dataclass(frozen=True)
class Node:
    """一个记忆节点。`obs` 与 `scene` **写入后不许改**（§14.3 冻结规则）。"""

    id: int
    kind: NodeKind
    summary: str
    scene: Optional[str] = None          # 语义层节点可以是 None（无场景锚）
    tags: tuple[str, ...] = ()
    obs: Observation = field(default_factory=Observation)
    modulators: Modulators = field(default_factory=Modulators)
    created_at: float = 0.0              # 逻辑时间（"天"）；由调用方给 ⇒ 可复现
    base_persist: float = 1.0            # §13.3 类型先验（event 快 / preference 慢 / reflection 最慢）
    s0: float = 0.0                      # 写入时算一次的显著度（§16.7-①）
    access_count: int = 0
    last_access: float = 0.0
    # §27（早期，调用方提案）：**复述累积的使用强度**。
    # ⚠️ 它与 `s0` **不是一回事**，这是本设计里最容易搞混的一处：
    #   * `s0`  = "**这件事当时有多重**"，由观测量算一次，**永久冻结**（§14.1/§14.3）
    #     —— 冻结保护的是**事实**，防的是"复述把情绪值推高"（越提越生气）；
    #   * `s_use` = "**她后来把它想起来过多少次、并被巩固了多少**"，**活的**，
    #     与 `access_count` 同类（§14.3 那张活量表），由复述累加、**增幅逐级递减**。
    #   ⇒ 有效显著度 `s_eff = s0 + s_use`；观测量的冻结**一个字都不用破**。
    s_use: float = 0.0
    source_quote: str = ""               # 出处（§3.1：对不上原文的条目一律丢弃）
    source_turn: str = ""
    obs_source: str = ""                 # 观测量**是谁打的**（rule / external / human / …）
    #   调用方可以按来源过滤节点。


@dataclass(frozen=True)
class Edge:
    """一条边。`weight` 会被 §14.5 的两种强化改 —— 那是**唯一**允许改边的地方。"""

    src: int
    dst: int
    kind: EdgeKind = EdgeKind.CO_OCCUR
    weight: float = 1.0
    decay_rate: float = 0.0     # ⚠️ §14.5：**必须有**，否则边会饱和、图退化成全连接
    evidence_quote: str = ""    # 出处（LLM 建链时必须给，对不上就丢）


@dataclass(frozen=True)
class GraphSnapshot:
    """**只读**图快照。检索只在快照上跑 ⇒ 同快照 + 同参数 = 逐位相同（§16.5）。

    `adj` 的出边顺序由 `(weight↓, dst↑)` 决定 —— 确定性排序是"可复现"的物理保证，
    不靠"实现的时候小心一点"。
    """

    nodes: Mapping[int, Node]
    edges: tuple[Edge, ...]
    adj: Mapping[int, tuple[Edge, ...]]

    def scene_of(self, node_id: int) -> Optional[str]:
        n = self.nodes.get(node_id)
        return None if n is None else n.scene

    def kind_of(self, node_id: int) -> Optional[NodeKind]:
        n = self.nodes.get(node_id)
        return None if n is None else n.kind

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def out_degree_normalized(edges: Iterable[Edge]) -> dict[tuple[int, int, str], float]:
    """`ŵ(i,j) = w(i,j) / Σ_k w(i,k)`（§15.2-①：出度归一，防热门节点天然放大）。

    🔴 早期 修（代码审查指出）：key 必须是 **`(src, dst, kind)`**，不是 `(src, dst)`。
    `edges` 表的唯一键是 `(src,dst,kind)` ⇒ **同一对节点可以同时有 `co_occur` 和 `derived_from`**
    （测试里就造过这种平行边）。用 `(src,dst)` 当 key 会让后写覆盖先写、两条边拿到同一个 ŵ，
    而分母把两条边的权重都算了进去 —— 一条边的传输量被另一条边的权重污染。
    """

    total: dict[int, float] = {}
    for e in edges:
        total[e.src] = total.get(e.src, 0.0) + max(e.weight, 0.0)
    out: dict[tuple[int, int, str], float] = {}
    for e in edges:
        denom = total.get(e.src, 0.0)
        out[(e.src, e.dst, e.kind.value)] = (max(e.weight, 0.0) / denom) if denom > 0 else 0.0
    return out
