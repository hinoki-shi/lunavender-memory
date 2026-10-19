#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成记忆库（**实验用，不是产品代码**）。

为什么需要它：§15.4 的"爆炸测试"要把记忆库从 100 条放大到 1000 条，而真实记忆库还没有。
合成库的作用是**在没真数据时先把结构性质问出来**（激活集合随规模怎么变、`D` 扫描单不单调）。

⚠️ 结论只对"这一族合成分布"成立。要推广到真实记忆，得等真库 —— 这一点写进了实验报告。

**生成规则**（两条标签通道，这是早期实验调出来的，见设计说明）：

1. **本地标签**：池子大小 ∝ 节点数（`n/3`），每节点随 2–3 个 ⇒ 平均出度**不随规模漂移**。
   （第一版把池子开小了：120 个节点接出 4374 条边、平均出度 33，扩散全被种子淹没。）
2. **共享实体**（薯条/咖啡/地铁/耳机）：**固定的一小撮**节点才带 ⇒ 跨场景共现是低频事件，
   也正是 §17 那个"麦当劳 ⊗ 鸟"的来源。要复现它，就把两条通道接上的那两个节点找出来。
"""

from __future__ import annotations

import random
from typing import Optional

from .ingest import reflect, remember
from .store import MemoryStore
from .types import Modulators, Observation

# 跨场景共享的实体（§17 的例子就是靠"薯条"这条边成立的）
SHARED_ENTITIES: tuple[str, ...] = ("薯条", "咖啡", "地铁", "耳机")
# **枢纽实体**：日常生活里少数几个实体连着大多数人（我/她/家/饭…）。
# 这就是 §6/§15.2 说的"超级节点"风险来源 —— 记忆越多，它们周围越挤。
HUB_ENTITIES: tuple[str, ...] = ("我", "她", "家", "饭", "学校", "手机", "睡", "今天")
TOPICS: tuple[str, ...] = ("吃", "学", "玩", "聊", "想", "累", "开心", "烦")


def build_bank(
    store: MemoryStore,
    *,
    n_nodes: int,
    seed: int = 7,
    semantic_fraction: float = 0.10,
    hot_fraction: float = 0.15,
    scenes: Optional[int] = None,
    tags_per_node: tuple[int, int] = (2, 3),
    real_entity_fraction: float = 0.06,
    hub_fraction: float = 0.35,
    hubs_per_node: tuple[int, int] = (1, 2),
    days: int = 30,
    params=None,
) -> dict:
    """造一个合成记忆库。返回统计 + 一个建议的查询词（`sample_query`）。"""

    rng = random.Random(seed)
    n_scenes = scenes if scenes else max(4, n_nodes // 25)
    scene_names = [f"scene{i:03d}" for i in range(n_scenes)]
    pool_size = max(8, n_nodes // 3)
    local_pool = [f"ent{i:05d}" for i in range(pool_size)]
    n_real = max(3, int(n_nodes * real_entity_fraction))
    real_slots = [SHARED_ENTITIES[i % len(SHARED_ENTITIES)] for i in range(n_real)]
    rng.shuffle(real_slots)
    by_scene: dict[str, list[int]] = {s: [] for s in scene_names}
    suggest_term = ""

    for i in range(n_nodes):
        scene = scene_names[i % n_scenes]
        k = rng.randint(*tags_per_node)
        tags = set(rng.sample(local_pool, k=k))
        if real_slots and rng.random() < 0.7:
            real = real_slots.pop()
            tags.add(real)
            if abs(i - n_nodes // 2) < 2:
                suggest_term = real
        if rng.random() < hub_fraction:
            kh = rng.randint(*hubs_per_node)
            tags.update(rng.sample(HUB_ENTITIES, k=min(kh, len(HUB_ENTITIES))))
        tags = tuple(sorted(tags))
        if not suggest_term and abs(i - n_nodes // 2) < 2 and tags:
            suggest_term = tags[0]
        hot = rng.random() < hot_fraction
        first_time = hot and rng.random() < 0.6
        obs = Observation(
            happy=rng.uniform(0.0, 1.0) if hot else rng.uniform(0.0, 0.35),
            angry=rng.uniform(0.0, 0.8) if hot and rng.random() < 0.3 else 0.0,
            shame=rng.uniform(0.0, 0.6) if hot and rng.random() < 0.2 else 0.0,
            # ⚠️ 早期 修：原来只写 `anticipation`（v4 起它不进 CHANNELS）⇒
            #    合成库的 hope/worry 恒为 0，D 扫描/爆炸测试是在"没有盼望也没有担心"的库上跑的。
            hope=rng.uniform(0.0, 0.9) if hot else rng.uniform(0.0, 0.3),
            worry=rng.uniform(0.0, 0.8) if hot and rng.random() < 0.35 else rng.uniform(0.0, 0.2),
            arousal=rng.uniform(0.5, 1.0) if hot else rng.uniform(0.0, 0.3),
            anxiety=rng.uniform(0.0, 1.0),
            importance=rng.uniform(0.5, 1.0) if hot else rng.uniform(0.0, 0.5),
            uncertainty=rng.uniform(0.0, 0.2),
        )
        node = remember(
            store,
            summary=f"{scene} 的{tags[0] if tags else '事'} #{i}",
            obs=obs,
            scene=scene,
            tags=tags,
            modulators=Modulators(first_time=first_time),
            created_at=float(rng.randint(0, days)),
            persist_key="preference" if rng.random() < 0.1 else "event",
            source_quote=f"[synth] {scene} #{i}",
            params=params,
        )
        by_scene[scene].append(node.id)

    # 语义层节点（抽象）：只能由反思新增，且**只能**靠 derived_from 跨场景（§9.3 / §15.1③）
    n_sem = int(n_nodes * semantic_fraction)
    ids_by_scene = {s: list(v) for s, v in by_scene.items()}
    for j in range(n_sem):
        picks = rng.sample(scene_names, k=min(2, len(scene_names)))
        source_ids: list[int] = []
        for s in picks:
            if ids_by_scene[s]:
                source_ids.append(rng.choice(ids_by_scene[s]))
        if not source_ids:
            continue
        nodes = {n.id: n for n in store.nodes()}
        sources = [nodes[i] for i in source_ids if i in nodes]
        if not sources:
            continue
        obs = Observation(happy=0.4, hope=0.3, worry=0.1, arousal=0.2, importance=0.8, anxiety=0.3)
        label = SHARED_ENTITIES[j % len(SHARED_ENTITIES)]
        reflect(
            store,
            summary=f"抽象 #{j}：他好像对「{label}」这类事有一套",
            sources=sources,
            obs=obs,
            created_at=float(rng.randint(0, days)),
            params=params,
        )

    snap = store.snapshot()
    return {
        "nodes": snap.n_nodes,
        "edges": snap.n_edges,
        "scenes": n_scenes,
        "avg_out_degree": round(len(snap.edges) / max(1, snap.n_nodes), 3),
        "semantic_nodes": sum(1 for n in snap.nodes.values() if n.scene is None),
        "seed": seed,
        "sample_query": suggest_term or (sorted(snap.nodes.values(), key=lambda n: n.id)[0].tags or ("",))[0],
    }
