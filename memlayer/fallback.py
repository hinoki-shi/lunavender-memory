#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""保守的向量兜底：主检索失败时只给“候选/不确定”，不直接当真。

主检索（词面 + 图扩散）仍然是第一选择。这个模块只在两种情况下工作：

1. 主检索没有种子，或种子分低于门槛；
2. 有种子，但种子对查询内容词的覆盖率太低（典型的单字巧合命中）。

向量命中要通过两道门：

* 分数/领先幅度都够高 ⇒ `use_vector`（作为“可能的记忆线索”，提示里仍要求核对）；
* 只在灰区 ⇒ `ask`（让调用方反问，不把候选写进事实性 prompt）；
* 其余 ⇒ `abstain`（宁可说没想起来，也不注入错误记忆）。

本模块不写主网、不改边权、不新增节点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .lexicon import tokenize
from .retrieval import RetrieveResult
from .types import GraphSnapshot
from .vectors import Embedder, VectorSidecar

_STOP = set(
    "的了在是我你他她它这一有不就都也很把被和跟与或吗呢啊吧哦嘛么个些没还又再会让给到过太"
    "好多多少什怎什么怎样呀啦对从向当只才已应该能会想说要去看做发生在后前时回上下里外中"
)


@dataclass(frozen=True)
class FallbackPolicy:
    """保守默认值； thresholds 必须用留出集校准，不要当普适常数。"""

    enabled: bool = True
    min_primary_seed: float = 0.05
    min_primary_coverage: float = 0.50
    top_k: int = 8
    max_candidates: int = 2
    # 无锚点时的接受线
    accept_score: float = 0.45
    accept_margin: float = 0.06
    # 有实体/标签锚点时可放宽
    anchor_accept_score: float = 0.38
    anchor_accept_margin: float = 0.02
    # 灰区：只反问，不注入
    review_score: float = 0.35
    review_margin: float = 0.01


@dataclass(frozen=True)
class FallbackResult:
    decision: str                    # primary / use_vector / ask / abstain
    reason: str
    candidates: tuple[int, ...] = ()
    scores: Mapping[int, float] = field(default_factory=dict)
    anchors: int = 0
    exclusive_anchors: int = 0
    margin: float = 0.0
    hint: str = ""

    @property
    def is_confident(self) -> bool:
        return self.decision in ("primary", "use_vector")


def content_terms(text: str) -> tuple[str, ...]:
    return tuple(t for t in tokenize(text) if t not in _STOP)


def _coverage(snapshot: GraphSnapshot, terms: Sequence[str], node_id: int) -> float:
    if not terms:
        return 0.0
    node = snapshot.nodes.get(int(node_id))
    if node is None:
        return 0.0
    node_terms = set(node.tags) | set(tokenize(node.summary))
    return len(set(terms) & node_terms) / len(set(terms))


def _anchor_count(snapshot: GraphSnapshot, terms: Sequence[str], node_id: int) -> int:
    node = snapshot.nodes.get(int(node_id))
    if node is None:
        return 0
    node_terms = set(node.tags) | set(tokenize(node.summary))
    return len(set(terms) & node_terms)


def _node_terms(snapshot: GraphSnapshot, node_id: int) -> set[str]:
    node = snapshot.nodes.get(int(node_id))
    if node is None:
        return set()
    return set(node.tags) | set(tokenize(node.summary))


def _exclusive_anchor_count(snapshot: GraphSnapshot, terms: Sequence[str],
                            top_id: int, second_id: Optional[int]) -> int:
    """top1 有、top2 没有的查询锚点；共享锚点不能用来“证明”唯一性。"""

    if not terms or second_id is None:
        return 0
    q = set(terms)
    top = q & _node_terms(snapshot, top_id)
    second = q & _node_terms(snapshot, second_id)
    return len(top - second)


class VectorFallback:
    def __init__(self, sidecar: Optional[VectorSidecar],
                 embedder: Optional[Embedder],
                 policy: Optional[FallbackPolicy] = None) -> None:
        self.sidecar = sidecar
        self.embedder = embedder
        self.policy = policy or FallbackPolicy()

    def decide(self, query_text: str, primary: RetrieveResult,
               snapshot: GraphSnapshot) -> FallbackResult:
        p = self.policy
        if not p.enabled:
            return FallbackResult("abstain", "disabled")

        terms = content_terms(query_text)
        primary_ok = False
        primary_reason = "no_primary_seed"
        if primary.seeds:
            top_seed = float(max(primary.seed_scores.values(), default=0.0))
            coverage = _coverage(snapshot, terms, primary.seeds[0])
            if top_seed >= p.min_primary_seed and coverage >= p.min_primary_coverage:
                primary_ok = True
            elif top_seed < p.min_primary_seed:
                primary_reason = "primary_seed_below_threshold"
            else:
                primary_reason = "primary_low_coverage"
        if primary_ok:
            return FallbackResult("primary", "primary_confident")

        if self.sidecar is None or self.embedder is None:
            return FallbackResult("abstain", primary_reason + ";no_vector")
        hits = self.sidecar.search(self.embedder.embed(query_text), k=p.top_k)
        if not hits:
            return FallbackResult("abstain", primary_reason + ";no_vector_hit")

        top_id, top_score = hits[0]
        second_id = hits[1][0] if len(hits) > 1 else None
        second_score = hits[1][1] if len(hits) > 1 else 0.0
        margin = float(top_score) - float(second_score)
        anchors = _anchor_count(snapshot, terms, top_id)
        exclusive = _exclusive_anchor_count(snapshot, terms, top_id, second_id)

        accept = (top_score >= p.accept_score and margin >= p.accept_margin)
        if exclusive > 0:
            accept = accept or (
                top_score >= p.anchor_accept_score
                and margin >= p.anchor_accept_margin
            )
        if accept:
            chosen = tuple(nid for nid, _ in hits[:max(1, p.max_candidates)])
            hint = self._render_use(snapshot, chosen)
            return FallbackResult("use_vector", "vector_accept", chosen,
                                  dict(hits[:len(chosen)]), anchors, exclusive,
                                  margin, hint)

        if top_score >= p.review_score and margin >= p.review_margin:
            chosen = tuple(nid for nid, _ in hits[:max(1, p.max_candidates)])
            hint = self._render_ask(snapshot, chosen)
            return FallbackResult("ask", "vector_gray_zone", chosen,
                                  dict(hits[:len(chosen)]), anchors, exclusive,
                                  margin, hint)

        return FallbackResult("abstain", "vector_below_threshold", (top_id,),
                              {top_id: float(top_score)}, anchors, exclusive,
                              margin, "")

    @staticmethod
    def _line(snapshot: GraphSnapshot, nid: int) -> str:
        node = snapshot.nodes.get(int(nid))
        if node is None:
            return ""
        tags = ("；标签：" + "、".join(node.tags)) if node.tags else ""
        return " - %s%s" % (node.summary, tags)

    def _render_use(self, snapshot: GraphSnapshot, ids: Sequence[int]) -> str:
        body = "\n".join(self._line(snapshot, nid) for nid in ids if nid in snapshot.nodes)
        return (
            "【可能的记忆线索（请核对，不要当作确定事实）】\n"
            + body
            + "\n如果线索对不上，请直接说“我可能记混了”，不要补写细节。"
        )

    def _render_ask(self, snapshot: GraphSnapshot, ids: Sequence[int]) -> str:
        body = "\n".join(self._line(snapshot, nid) for nid in ids if nid in snapshot.nodes)
        return (
            "【记忆不确定】我只想起一些相近的事：\n"
            + body
            + "\n先向用户确认是哪一件，不要把候选当成事实。"
        )
