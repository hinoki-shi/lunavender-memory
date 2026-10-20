#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可重建的向量检索 sidecar（实验原型，不改变主网）。

设计边界：

* 主网 `nodes / edges / tags` 仍是唯一事实源；
* 本模块只把节点的检索文本投影成向量，存在**独立 SQLite 文件**里；
* sidecar 丢了可以从主网全量重建，节点/边/权重一个字都不动；
* `HashEmbedder` 是确定性测试替身，不是语义模型；真实模型通过
  `Embedder` 协议在外面接入（例如 BGE-M3 / multilingual-e5）。

这不是“压缩主网”。它是“给主网加一个可丢弃、可重建的语义入口”。
"""

from __future__ import annotations

import array
import hashlib
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Protocol, Sequence

from .lexicon import tokenize
from .store import MemoryStore
from .types import Node

SCHEMA = """
CREATE TABLE IF NOT EXISTS node_vectors (
    node_id   INTEGER PRIMARY KEY,
    model     TEXT    NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB    NOT NULL,
    text_hash TEXT    NOT NULL,
    updated_at REAL   NOT NULL
);
"""


class Embedder(Protocol):
    """外部 embedding 模型要实现的最小接口。"""

    model: str
    dim: int

    def embed(self, text: str) -> tuple[float, ...]:
        ...


@dataclass(frozen=True)
class VectorRecord:
    node_id: int
    model: str
    dim: int
    vector: tuple[float, ...]
    text_hash: str
    updated_at: float


def retrieval_text(node: Node, *, include_tags: bool = False) -> str:
    """节点的可检索文本：默认摘要 + 场景 + 类型；标签不默认塞进向量。

    默认**不嵌 source_quote**：原文更适合留在主网做审计，不应被复制到
    另一个索引文件里；如果未来要加，必须同步纳入删除/隐私边界。

    标签只作为词面/图锚点；真实数据的标签往往有噪声，直接拼进 embedding
    会稀释语义（实测会把向量 R@1 从 .913 拉低到 .870）。
    """

    parts = [str(node.summary or "")]
    if include_tags and node.tags:
        parts.append(" ".join(sorted(str(t) for t in node.tags)))
    if node.scene:
        parts.append(str(node.scene))
    parts.append(node.kind.value)
    return "\n".join(p for p in parts if p)


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _normalise(vec: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 1e-12:
        return tuple(0.0 for _ in vec)
    return tuple(float(x) / norm for x in vec)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


class HashEmbedder:
    """确定性、零依赖的测试替身（**不是语义模型**）。

    它把 token 哈希到固定维度，用于验证 sidecar 的建库、增量同步、删除和
    检索管线；它只能做“同词相近”，不能替代真实 embedding。
    """

    def __init__(self, dim: int = 256, model: str = "hash-stub-v1") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.model = str(model)

    def embed(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        if not toks:
            vec[0] = 1.0
            return tuple(vec)
        for tok in toks:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            n = int.from_bytes(h, "little")
            bucket = n % self.dim
            sign = 1.0 if (n >> 8) & 1 else -1.0
            vec[bucket] += sign
        return _normalise(vec)


class VectorSidecar:
    """独立 SQLite 文件里的节点向量表。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA secure_delete = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        if self.path != ":memory:":
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    def close(self) -> None:
        self.conn.close()

    def clear(self) -> None:
        self.conn.execute("DELETE FROM node_vectors")
        self.conn.commit()

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*), MIN(model), MIN(dim) FROM node_vectors").fetchone()
        return {"records": int(row[0]), "model": row[1], "dim": row[2]}

    def records(self) -> list[VectorRecord]:
        rows = self.conn.execute(
            "SELECT node_id, model, dim, vector, text_hash, updated_at "
            "FROM node_vectors ORDER BY node_id").fetchall()
        out: list[VectorRecord] = []
        for node_id, model, dim, blob, th, ts in rows:
            vec = array.array("f")
            vec.frombytes(blob)
            out.append(VectorRecord(int(node_id), str(model), int(dim),
                                    tuple(float(x) for x in vec), str(th), float(ts)))
        return out

    def sync(self, store: MemoryStore, embedder: Embedder, *,
             force: bool = False, now: float = 0.0) -> dict:
        """把主网节点同步进 sidecar；只重算新增/变更/换模型的节点。"""

        nodes = list(store.nodes())
        existing = {r.node_id: r for r in self.records()}
        live_ids = {n.id for n in nodes}
        embedded = reused = 0
        for node in nodes:
            text = retrieval_text(node)
            h = text_hash(text)
            old = existing.get(node.id)
            if (not force and old is not None
                    and old.model == embedder.model
                    and old.dim == embedder.dim
                    and old.text_hash == h):
                reused += 1
                continue
            vec = tuple(float(x) for x in embedder.embed(text))
            if len(vec) != embedder.dim:
                raise ValueError(
                    f"embedder returned {len(vec)} dims, expected {embedder.dim}")
            self.conn.execute(
                "INSERT INTO node_vectors "
                "(node_id, model, dim, vector, text_hash, updated_at) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(node_id) DO UPDATE SET "
                "model=excluded.model, dim=excluded.dim, vector=excluded.vector, "
                "text_hash=excluded.text_hash, updated_at=excluded.updated_at",
                (node.id, embedder.model, embedder.dim,
                 array.array("f", vec).tobytes(), h, float(now)),
            )
            embedded += 1
        stale = [nid for nid in existing if nid not in live_ids]
        if stale:
            q = ",".join("?" for _ in stale)
            self.conn.execute(f"DELETE FROM node_vectors WHERE node_id IN ({q})", stale)
        self.conn.commit()
        return {"embedded": embedded, "reused": reused, "deleted": len(stale)}

    def delete(self, node_ids: Iterable[int]) -> int:
        ids = sorted({int(i) for i in node_ids})
        if not ids:
            return 0
        q = ",".join("?" for _ in ids)
        cur = self.conn.execute(
            f"DELETE FROM node_vectors WHERE node_id IN ({q})", ids)
        self.conn.commit()
        return int(cur.rowcount)

    def upsert(self, node_id: int, vector: Sequence[float], *,
               model: str, text_hash_value: str = "",
               updated_at: float = 0.0) -> None:
        """直接写入一个向量（给外部 embedder/测试用；常规路径用 sync）。"""

        vec = _normalise(tuple(float(x) for x in vector))
        if not vec:
            raise ValueError("vector must not be empty")
        self.conn.execute(
            "INSERT INTO node_vectors "
            "(node_id, model, dim, vector, text_hash, updated_at) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(node_id) DO UPDATE SET "
            "model=excluded.model, dim=excluded.dim, vector=excluded.vector, "
            "text_hash=excluded.text_hash, updated_at=excluded.updated_at",
            (int(node_id), str(model), len(vec),
             array.array("f", vec).tobytes(),
             str(text_hash_value), float(updated_at)),
        )
        self.conn.commit()

    def search(self, query_vector: Sequence[float], k: int = 10, *,
               allowed_ids: Optional[Iterable[int]] = None,
               model: Optional[str] = None) -> list[tuple[int, float]]:
        """余弦检索；向量在写入时已归一化，所以点积即余弦。"""

        qv = _normalise(tuple(float(x) for x in query_vector))
        allowed = {int(i) for i in allowed_ids} if allowed_ids is not None else None
        hits: list[tuple[int, float]] = []
        for rec in self.records():
            if model is not None and rec.model != model:
                continue
            if allowed is not None and rec.node_id not in allowed:
                continue
            if len(rec.vector) != len(qv):
                continue
            score = _dot(qv, rec.vector)
            if score > 0:
                hits.append((rec.node_id, score))
        hits.sort(key=lambda kv: (-kv[1], kv[0]))
        return hits[:max(0, int(k))]


def rrf_fuse(rankings: Sequence[Sequence[int]], *, k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion：把词面/向量的名次融合成可比分数。"""

    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, nid in enumerate(dict.fromkeys(int(x) for x in ranking), 1):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])))


def ranked_ids(scores: Mapping[int, float]) -> list[int]:
    return [int(nid) for nid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def hybrid_seed_scores(
    lexical: Mapping[int, float],
    vector_hits: Sequence[tuple[int, float]],
    *,
    limit: int = 20,
    rrf_k: int = 60,
) -> dict[int, float]:
    """词面 + 向量名次融合后截断成 seed 分。"""

    fused = rrf_fuse(
        [ranked_ids(lexical), [int(nid) for nid, _ in vector_hits]],
        k=rrf_k,
    )
    if limit <= 0:
        return {}
    top = max(fused.values(), default=0.0)
    if top <= 0:
        return {}
    # 归一化后满足 retrieve() 的 seed_threshold；RRF 原始分数只在排序时有意义。
    return {nid: score / top for nid, score in list(fused.items())[:limit]}
