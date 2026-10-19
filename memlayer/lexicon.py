#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段一的**词面相关**（§3.2 的"BM25 + 向量"里最便宜的那一半）。

⚠️ 这个模块是**可替换的**：真上向量召回时，只要换一个签名相同的函数
（`(snapshot, terms, scene) -> {node_id: 已归一分数}`），检索层不用改。
这就是"不把检索和召回实现焊死"的意思。
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional, Sequence

from .types import GraphSnapshot

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> tuple[str, ...]:
    """极简分词：英文/数字按词，中文**按字**（够用于实验；真上生产换 jieba/BM25）。"""

    return tuple(t.lower() for t in _TOKEN_RE.findall(text or ""))


def lexical_relevance(
    snapshot: GraphSnapshot,
    terms: Sequence[str],
    scene: Optional[str] = None,
    *,
    scene_bonus: float = 0.15,
) -> dict[int, float]:
    """词面重合度 ∈ [0, 1]（按最大分归一）。`scene` 给一点加成（场景锚也算线索）。

    返回 `{node_id: score}`，只含分数 > 0 的节点 —— 阶段一的"候选"。
    """

    q = {t.lower() for t in terms if t}
    if not q:
        return {}
    raw: dict[int, float] = {}
    for nid, node in snapshot.nodes.items():
        node_terms = set(node.tags) | set(tokenize(node.summary))
        overlap = len(q & node_terms)
        if overlap <= 0:
            continue
        score = overlap / len(q)
        if scene is not None and node.scene == scene:
            score += scene_bonus
        raw[nid] = score
    if not raw:
        return {}
    top = max(raw.values())
    return {k: (v / top if top > 0 else 0.0) for k, v in raw.items()}


def z_normalize(values: Mapping[int, float]) -> dict[int, float]:
    """把一组分数**拉到同一量纲**（§16.7-⑥：种子公式里各项必须先归一，不能等权相加）。

    用 min-max 而不是 z-score：输出稳定落在 [0,1]，且不需要假设分布。
    """

    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    span = hi - lo
    if span <= 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def coverage(values: Iterable[float]) -> float:
    """辅助：一列分数里有多少是正的（实验报告用）。"""

    vals = list(values)
    if not vals:
        return 0.0
    return sum(1 for v in vals if v > 0) / len(vals)
