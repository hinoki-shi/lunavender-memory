#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""写入路径（§14.1 那条 DAG：**一次算完、落盘、之后冻结**）。

为什么单独一个模块：存储层只做 I/O、不碰数学（防耦合），
于是"算 S₀ → 写节点 → 建规则链"这条链必须有个明确的家 —— 就是这里。

⚠️ 三条纪律：
1. **观测量写进去就冻结**：本模块没有任何"更新 obs"的入口（§14.3）；
2. **规则建链照建**：共享实体/标签的边**建出来是对的**（那是事实）；
   能不能"走"由检索侧的边类型规则决定（§15.1③）—— 建链和能不能走是两件事；
3. **抽象只能新增**：`reflect()` 永远新建语义层节点 + `derived_from` 边指向来源，
   **不改任何旧节点**（§9.3）。
"""

from __future__ import annotations

from typing import Optional, Sequence

from .config import Params
from .privacy import looks_sensitive, reject_if_sensitive
from .store import MemoryStore
from .types import EdgeKind, Modulators, Node, NodeKind, Observation
from .weights import salience0

# 类型先验（§13.3：类型决定"起点"，强度决定"加成"）
BASE_PERSIST: dict[str, float] = {
    "event": 1.0,        # 一次性事件：忘得快
    "preference": 3.0,   # 偏好：活很久（"他不喜欢被劝早睡"哪怕当时情绪不强）
    "reflection": 6.0,   # 反思/抽象：最慢
}


def remember(
    store: MemoryStore,
    *,
    summary: str,
    obs: Observation,
    params: Optional[Params] = None,
    scene: Optional[str] = None,
    tags: Sequence[str] = (),
    kind: NodeKind = NodeKind.EPISODE,
    modulators: Optional[Modulators] = None,
    created_at: float = 0.0,
    persist_key: str = "event",
    source_quote: str = "",
    source_turn: str = "",
    obs_source: str = "rule",
    link: bool = True,
    reject_sensitive: bool = True,
) -> Node:
    """写一条记忆。`S₀` 在这里算一次（§16.7-① 的 `S₀`），之后**永不重算**。

    `reject_sensitive`：摘要 / 原话 / 标签疑似凭据 ⇒ 抛 `SensitiveContentError`，**不落盘**。
    """

    if reject_sensitive:
        reject_if_sensitive(
            summary, scene or "", source_quote, source_turn, obs_source,
            " ".join(tags), label="记忆写入",
        )
    p = params or Params()
    mods = modulators or Modulators()
    s0 = salience0(obs, mods, p)
    base_persist = BASE_PERSIST.get(persist_key, 1.0)
    node = store.add_node(
        kind=kind,
        summary=summary,
        scene=scene,
        tags=tuple(tags),
        obs=obs,
        modulators=mods,
        created_at=created_at,
        s0=s0,
        base_persist=base_persist,
        source_quote=source_quote,
        source_turn=source_turn,
        obs_source=obs_source,
        reject_sensitive=reject_sensitive,
    )
    if link:
        link_by_tags(store, node, params=p)
    return node


def link_by_tags(store: MemoryStore, node: Node, *, params: Optional[Params] = None) -> int:
    """规则建链：与已有节点共享标签 ⇒ 建 `co_occur` 边（双向，权重 = 共享标签数）。

    这是 §3.1 的"规则建链（便宜、可靠）"。边**照建**，
    但"能不能跨场景走"是检索侧的事（§15.1③）—— 别把两件事混在一个地方。
    """

    p = params or Params()
    mine = set(node.tags)
    if not mine:
        return 0
    candidate_ids = [i for i in store.node_ids_with_any_tag(tuple(mine)) if i != node.id]
    if not candidate_ids:
        return 0
    created = 0
    for other in store.nodes_by_ids(candidate_ids):
        shared = mine & set(other.tags)
        if not shared:
            continue
        w = float(len(shared))
        store.add_edge(
            src=node.id,
            dst=other.id,
            kind=EdgeKind.CO_OCCUR,
            weight=w,
            decay_rate=p.edge_decay_rate,
        )
        store.add_edge(
            src=other.id,
            dst=node.id,
            kind=EdgeKind.CO_OCCUR,
            weight=w,
            decay_rate=p.edge_decay_rate,
        )
        created += 2
    return created


def reflect(
    store: MemoryStore,
    *,
    summary: str,
    sources: Sequence[Node],
    obs: Optional[Observation] = None,
    params: Optional[Params] = None,
    created_at: float = 0.0,
    source_quote: str = "",
    reject_sensitive: bool = True,
) -> Node:
    """反思：**新增**一个语义层节点，并从每个来源建一条 `derived_from` 边。

    🔑 这是 §9.3 说的那条路，也是**唯一**的跨场景通道（§15.1③）：
    "麦当劳那次"和"游泳那次"不许被揉成一句，但可以**共同派生**出一个抽象节点。
    """

    if reject_sensitive:
        reject_if_sensitive(summary, source_quote, label="反思写入")
        for src in sources:
            reject_if_sensitive(src.source_quote, label="反思证据")
    p = params or Params()
    node = store.add_node(
        kind=NodeKind.SEMANTIC,
        summary=summary,
        scene=None,  # 抽象没有场景锚（§9.3）
        tags=(),
        obs=obs or Observation(),
        modulators=Modulators(),
        created_at=created_at,
        s0=salience0(obs or Observation(), Modulators(), p),
        base_persist=BASE_PERSIST["reflection"],
        source_quote=source_quote,
        source_turn="",
        obs_source="rule",
        reject_sensitive=reject_sensitive,
    )
    for src in sources:
        store.add_edge(
            src=src.id,
            dst=node.id,
            kind=EdgeKind.DERIVED_FROM,
            weight=1.0,
            decay_rate=p.edge_decay_rate,
            evidence_quote=src.source_quote,
        )
    return node


def induce_preferences(
    store: MemoryStore,
    *,
    min_events: int = 2,
    positive_at: float = 0.5,
    negative_at: float = 0.5,
    max_angry: float = 0.3,
    params: Optional[Params] = None,
    created_at: float = 0.0,
) -> list[Node]:
    """从**重复出现的事件**里归纳出偏好节点（**规则版 v1**）。

    🔑 为什么需要它（调用方 早期 的例子）：

    > 「薯条脆可能把『刚炸很好吃』这个作为长期记忆」

    但那条事件本身**确实只是"今天的一件事"**（实测：机器问"这段文本是不是关于他的偏好"只给 0.41）——
    真正该长期留存的是从中归纳出的**认知**：「他喜欢刚炸的薯条」。
    这正是 §9.3 的规矩：**抽象只能新增节点，不能把情景节点改判成别的类型**。

    **v1 规则**（先跑通、可复现；LLM 版留给 §12.2 的 L1）：
    找出被 ≥`min_events` 条情景节点共享的标签（如"薯条"），若这些事件的**情绪一致偏正**
    （`happy` 均值 ≥ `positive_at` 且 `angry` 均值 ≤ `max_angry`），
    就新增一个 `preference` 节点「他好像挺喜欢「薯条」这类东西」+ 从那些事件建 `derived_from` 边。
    负性一致（`angry` 均值 ≥ `negative_at`）同理生成"不太喜欢"。

    **幂等**：已有同标签的偏好节点就跳过（重复跑不会堆节点）。
    """

    p = params or Params()
    created: list[Node] = []
    episodes = [n for n in store.nodes() if n.kind is NodeKind.EPISODE and n.tags]
    by_tag: dict[str, list[Node]] = {}
    for node in episodes:
        for tag in node.tags:
            by_tag.setdefault(tag, []).append(node)

    existing = {n.summary for n in store.nodes() if n.kind is NodeKind.SEMANTIC}
    for tag in sorted(by_tag):
        group = by_tag[tag]
        if len(group) < min_events:
            continue
        happy = sum(n.obs.happy for n in group) / len(group)
        angry = sum(n.obs.angry for n in group) / len(group)
        if happy >= positive_at and angry <= max_angry:
            summary = f"他好像挺喜欢「{tag}」这类东西"
        elif angry >= negative_at:
            summary = f"他好像不太喜欢「{tag}」这类东西"
        else:
            continue          # 情绪不一致 ⇒ 不足以归纳（不敢乱下结论）
        if summary in existing:
            continue
        obs = Observation(happy=max(0.0, happy), angry=max(0.0, angry), importance=0.6,
                          arousal=0.2, unfinished=0.0)
        quote = "；".join(n.source_quote for n in group[:3] if n.source_quote)[:200]
        if looks_sensitive(summary) or looks_sensitive(quote):
            continue
        node = store.add_node(
            kind=NodeKind.SEMANTIC,
            summary=summary,
            scene=None,                      # 偏好没有场景锚（§9.3）
            tags=(tag,),
            obs=obs,
            modulators=Modulators(),
            created_at=created_at,
            s0=salience0(obs, Modulators(), p),
            base_persist=BASE_PERSIST["preference"],   # ← "偏好活得久"是**类型先验**给的
            source_quote=quote,
            obs_source="rule",
        )
        for src in group:
            store.add_edge(src=src.id, dst=node.id, kind=EdgeKind.DERIVED_FROM, weight=1.0,
                           decay_rate=p.edge_decay_rate, evidence_quote=src.source_quote)
        created.append(node)
    return created
