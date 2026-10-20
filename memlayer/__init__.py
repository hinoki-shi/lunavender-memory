#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆层（图 + 衰减 + 发散度）。

**只导出稳定 API**：内部模块可以随便重构，但这里导出的名字不动 ——
这是"上层不被内部重构牵动"的护栏（防耦合）。
"""

from __future__ import annotations

from .config import Params, params_for_path
from .divergence import DParams, params_for
from .ingest import BASE_PERSIST, induce_preferences, reflect, remember
from .lexicon import lexical_relevance, tokenize
from .lifecycle import NodeView, advance_time, commit_access, freeze_signature, recency_of, view_of
from .retrieval import (
    Query,
    RetrieveResult,
    SceneSetResult,
    edge_allowed,
    retrieve,
    retrieve_scene_set,
    scene_candidates,
    random_recall,
    random_seed_nodes,
)
from .privacy import (
    SensitiveContentError,
    find_text_in_files,
    looks_sensitive,
    reject_if_sensitive,
    scrub_files,
)
from .reflection_gate import (
    ReflectionCandidate,
    ReflectionDecision,
    ReflectionGate,
    ReflectionPolicy,
)
from .selftrace import TRACE_KINDS, SelfTrace, activation_fingerprint, params_fingerprint
from .store import MemoryStore, snapshot_from_rows
from .types import (
    CHANNELS,
    Edge,
    EdgeKind,
    GraphSnapshot,
    Modulators,
    MemoryState,
    Node,
    NodeKind,
    Observation,
)
from .vectors import (
    HashEmbedder,
    VectorRecord,
    VectorSidecar,
    hybrid_seed_scores,
    ranked_ids,
    retrieval_text,
    rrf_fuse,
    text_hash,
)
from .weights import (
    channel_vector,
    decay_rate,
    expressed_arousal,
    keep_ratio,
    persistence,
    s_now,
    salience0,
)

__all__ = [
    # 契约
    "CHANNELS",
    "Edge",
    "EdgeKind",
    "GraphSnapshot",
    "Modulators",
    "Node",
    "NodeKind",
    "MemoryState",
    "Observation",
    # 向量 sidecar（实验原型；不改变主网）
    "HashEmbedder",
    "VectorRecord",
    "VectorSidecar",
    "hybrid_seed_scores",
    "ranked_ids",
    "retrieval_text",
    "rrf_fuse",
    "text_hash",
    # 参数与发散度
    "Params",
    "params_for_path",
    "DParams",
    "params_for",
    # 权重
    "channel_vector",
    "decay_rate",
    "expressed_arousal",
    "keep_ratio",
    "persistence",
    "s_now",
    "salience0",
    # 时间 / 访问
    "NodeView",
    "advance_time",
    "commit_access",
    "freeze_signature",
    "recency_of",
    "view_of",
    # 检索
    "Query",
    "RetrieveResult",
    "SceneSetResult",
    "edge_allowed",
    "lexical_relevance",
    "retrieve",
    "retrieve_scene_set",
    "scene_candidates",
    "random_recall",
    "random_seed_nodes",
    "tokenize",
    # 写入路径
    "BASE_PERSIST",
    "induce_preferences",
    "reflect",
    "remember",
    # 存储
    "MemoryStore",
    "snapshot_from_rows",
    # 隐私
    "SensitiveContentError",
    "looks_sensitive",
    "reject_if_sensitive",
    "find_text_in_files",
    "scrub_files",
    # 自我轨迹（辅网 v0）
    "SelfTrace",
    "TRACE_KINDS",
    "params_fingerprint",
    "activation_fingerprint",
    # 主动反思三闸门
    "ReflectionCandidate",
    "ReflectionDecision",
    "ReflectionGate",
    "ReflectionPolicy",
]
