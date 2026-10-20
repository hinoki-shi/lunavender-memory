#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""存储层：**全项目唯一碰 SQL 的文件**（§3.1 的三张表）。

为什么只有这里碰 SQL：换存储（SQLite → sqlite-vec / 向量库 / 云端）时只改这一个文件，
检索和权重一行都不用动（防耦合的第二条硬规矩）。

⚠️ **这个类只做 I/O，不做数学**：
* 显著度 `S₀` 由 `ingest.py` 算好再写进来（写入路径的 DAG 在那边，§14.1）；
* 这个类**没有**任何修改 `obs` / `scene` / `s0` 的方法 —— 冻结规则（§14.3）
  是**结构上**保证的，不是靠"记得别改"。
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .privacy import looks_sensitive, reject_if_sensitive
from .types import (
    Edge,
    EdgeKind,
    GraphSnapshot,
    Modulators,
    Node,
    NodeKind,
    Observation,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id            INTEGER PRIMARY KEY,
    kind          TEXT    NOT NULL,
    summary       TEXT    NOT NULL,
    scene         TEXT,
    tags          TEXT    NOT NULL DEFAULT '[]',
    obs           TEXT    NOT NULL,
    modulators    TEXT    NOT NULL DEFAULT '{}',
    created_at    REAL    NOT NULL DEFAULT 0,
    base_persist  REAL    NOT NULL DEFAULT 1.0,
    s0            REAL    NOT NULL DEFAULT 0.0,
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_access   REAL    NOT NULL DEFAULT 0.0,
    s_use         REAL    NOT NULL DEFAULT 0.0,
    source_quote  TEXT    NOT NULL DEFAULT '',
    source_turn   TEXT    NOT NULL DEFAULT '',
    obs_source    TEXT    NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS edges (
    id             INTEGER PRIMARY KEY,
    src            INTEGER NOT NULL,
    dst            INTEGER NOT NULL,
    kind           TEXT    NOT NULL,
    weight         REAL    NOT NULL DEFAULT 1.0,
    decay_rate     REAL    NOT NULL DEFAULT 0.0,
    evidence_quote TEXT    NOT NULL DEFAULT '',
    UNIQUE(src, dst, kind)
);
CREATE TABLE IF NOT EXISTS tags (
    node_id INTEGER NOT NULL,
    tag     TEXT    NOT NULL,
    UNIQUE(node_id, tag)
);
CREATE TABLE IF NOT EXISTS tombstones (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id    INTEGER NOT NULL,
    kind       TEXT    NOT NULL,
    deleted_at REAL    NOT NULL,
    reason     TEXT    NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS self_trace (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL    NOT NULL,
    kind    TEXT    NOT NULL,
    payload TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_nodes_scene ON nodes(scene);
"""

SCHEMA_VERSION = 1


def _restrict_file(path: str | Path) -> None:
    """尽力把数据库/备份收紧到仅所有者可读写；Windows 上只是尽力而为。"""

    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _lastrowid(cur: sqlite3.Cursor) -> int:
    """SQLite 插入成功时必然有 rowid；把 `None` 变成明确错误，方便类型检查。"""

    row_id = cur.lastrowid
    if row_id is None:
        raise RuntimeError("SQLite insert returned no rowid")
    return int(row_id)


_NODE_ID_KEYS = frozenset({
    "produced", "rejected", "activation", "evidence", "node_id",
    "evidence_ids", "source_ids", "node_ids",
})


def _iter_node_ids(value: object):
    """只收集 payload 里**明确表示节点 id** 的字段，避免误扫 day_index 等整数。"""

    if isinstance(value, dict):
        for key, item in value.items():
            if key in _NODE_ID_KEYS:
                if isinstance(item, int) and not isinstance(item, bool):
                    yield item
                elif isinstance(item, (list, tuple)):
                    for one in item:
                        if isinstance(one, int) and not isinstance(one, bool):
                            yield one
            elif isinstance(item, (dict, list, tuple)):
                yield from _iter_node_ids(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_node_ids(item)


def _iter_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _atomic_backup(src_conn: sqlite3.Connection, dest: str | Path, *,
                   overwrite: bool = False) -> Path:
    dest_path = Path(dest).expanduser().resolve()
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"导出目标已存在：{dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + f".{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    tmp_path.touch(mode=0o600, exist_ok=True)
    _restrict_file(tmp_path)
    try:
        with closing(sqlite3.connect(tmp_path)) as dst:
            src_conn.commit()
            src_conn.backup(dst)
            integrity = str(dst.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                raise RuntimeError(f"导出快照未通过完整性检查：{integrity}")
            dst.commit()
        os.replace(tmp_path, dest_path)
        _restrict_file(dest_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return dest_path


def _obs_to_json(obs: Observation) -> str:
    return json.dumps(obs.as_dict(), ensure_ascii=False, sort_keys=True)


_OBS_FIELDS = frozenset(f.name for f in fields(Observation))


def _obs_from_json(raw: str) -> Observation:
    """**容忍未知键**（早期 加固）。

    以前是裸的 `Observation(**json.loads(raw))` ⇒ 只要改/删一个字段名，**整库读不出来**
    （代码审查第 13.4 条点过这条；`calm` → 派生属性时真的踩了一次）。
    现在只取 dataclass 真有的字段，别的一律忽略（旧行里的 `calm` 就这样被安全丢掉）。
    """

    data = json.loads(raw)
    if not isinstance(data, dict):
        return Observation()
    return Observation(**{k: v for k, v in data.items() if k in _OBS_FIELDS})


def _mods_to_json(mod: Modulators) -> str:
    return json.dumps(asdict(mod), ensure_ascii=False, sort_keys=True)


_MODS_FIELDS = frozenset(f.name for f in fields(Modulators))


def _mods_from_json(raw: str) -> Modulators:
    """**容忍未知键**：旧库/新库的调制器字段名变化时，别让整库读不出来。"""

    data = json.loads(raw)
    if not isinstance(data, dict):
        return Modulators()
    return Modulators(**{k: v for k, v in data.items() if k in _MODS_FIELDS})


class MemoryStore:
    """SQLite 版存储。`path=":memory:"` 用于测试与实验（不落盘）。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not self.path.startswith("file:"):
                Path(self.path).touch(mode=0o600, exist_ok=True)
                _restrict_file(self.path)
        self.conn = sqlite3.connect(self.path)
        if self.path != ":memory:":
            _restrict_file(self.path)
        # 删除要"真删"：被删行的页面用 0 覆盖，而不是只打删除标记（见 forget()）
        self.conn.execute("PRAGMA secure_delete = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """给**已经存在**的库补新列（§27 的 `s_use`）。

        教训（早期）：改字段名曾经让整库读不出来（`Observation(**json)` 那次）。
        所以新列一律"缺就补、有就跳过"，旧库能直接用。
        """

        have = {r[1] for r in self.conn.execute("PRAGMA table_info(nodes)").fetchall()}
        if "s_use" not in have:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN s_use REAL NOT NULL DEFAULT 0.0")

    def counts(self) -> dict[str, int]:
        """各表行数（备份清单/迁移验收用；不读正文）。"""

        return {
            "nodes": int(self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]),
            "edges": int(self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]),
            "tags": int(self.conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]),
            "tombstones": int(self.conn.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0]),
            "self_trace": int(self.conn.execute("SELECT COUNT(*) FROM self_trace").fetchone()[0]),
        }

    def export_snapshot(self, dest: str | Path, *, overwrite: bool = False) -> dict:
        """用 SQLite backup API 导出一致性整库快照。

        导出包含节点、边、标签、墓碑和自我轨迹；源库只读，不改任何业务行。
        先写同目录临时文件，校验后再原子替换，避免中途失败留下半个备份。
        """

        dest_path = Path(dest).expanduser().resolve()
        if self.path != ":memory:" and dest_path == Path(self.path).expanduser().resolve():
            raise ValueError("导出目标不能是源数据库本身")
        _atomic_backup(self.conn, dest_path, overwrite=overwrite)
        return {"path": str(dest_path), "counts": self.counts(), "integrity": "ok"}

    # ------------------------------------------------------------ 写
    def add_node(
        self,
        *,
        kind: NodeKind | str,
        summary: str,
        scene: Optional[str],
        tags: Sequence[str],
        obs: Observation,
        modulators: Modulators,
        created_at: float,
        s0: float,
        base_persist: float,
        source_quote: str = "",
        source_turn: str = "",
        obs_source: str = "",
        node_id: Optional[int] = None,
        reject_sensitive: bool = True,
    ) -> Node:
        """插入一个节点。`s0` 必须**由写入路径算好传进来**（见 `ingest.remember`）。

        `reject_sensitive=False` 只用于导入旧数据/测试；新写入不要关闭闸门。
        """

        if reject_sensitive:
            reject_if_sensitive(
                summary, scene or "", source_quote, source_turn, obs_source,
                " ".join(tags), label="节点写入",
            )
        kind_v = kind.value if isinstance(kind, NodeKind) else str(kind)
        cur = self.conn.execute(
            """INSERT INTO nodes
               (id, kind, summary, scene, tags, obs, modulators, created_at,
                base_persist, s0, access_count, last_access, source_quote, source_turn, obs_source)
               VALUES (?,?,?,?,?,?,?,?,?,?,0,0,?,?,?)""",
            (
                node_id,
                kind_v,
                summary,
                scene,
                json.dumps(list(tags), ensure_ascii=False),
                _obs_to_json(obs),
                _mods_to_json(modulators),
                float(created_at),
                float(base_persist),
                float(s0),
                source_quote,
                source_turn,
                obs_source,
            ),
        )
        self.conn.commit()
        row_id = _lastrowid(cur)
        for tag in tags:
            self.conn.execute(
                "INSERT OR IGNORE INTO tags (node_id, tag) VALUES (?,?)",
                (row_id, tag),
            )
        self.conn.commit()
        return Node(
            id=row_id,
            kind=NodeKind(kind_v),
            summary=summary,
            scene=scene,
            tags=tuple(tags),
            obs=obs,
            modulators=modulators,
            created_at=float(created_at),
            base_persist=float(base_persist),
            s0=float(s0),
            source_quote=source_quote,
            source_turn=source_turn,
            obs_source=obs_source,
        )

    def add_edge(
        self,
        *,
        src: int,
        dst: int,
        kind: EdgeKind | str,
        weight: float = 1.0,
        decay_rate: float = 0.0,
        evidence_quote: str = "",
        merge: bool = True,
        reject_sensitive: bool = True,
    ) -> None:
        """插一条边。`merge=True` 时同 (src,dst,kind) 已存在则**取较大权重**（幂等建链）。

        `reject_sensitive=False` 只用于导入旧数据/测试。
        """

        if reject_sensitive:
            reject_if_sensitive(evidence_quote, label="边证据")
        kind_v = kind.value if isinstance(kind, EdgeKind) else str(kind)
        if merge:
            row = self.conn.execute(
                "SELECT id, weight FROM edges WHERE src=? AND dst=? AND kind=?",
                (src, dst, kind_v),
            ).fetchone()
            if row is not None:
                new_w = max(float(row[1]), float(weight))
                self.conn.execute("UPDATE edges SET weight=? WHERE id=?", (new_w, row[0]))
                self.conn.commit()
                return
        self.conn.execute(
            """INSERT OR IGNORE INTO edges (src, dst, kind, weight, decay_rate, evidence_quote)
               VALUES (?,?,?,?,?,?)""",
            (src, dst, kind_v, float(weight), float(decay_rate), evidence_quote),
        )
        self.conn.commit()

    # ------------------------------------------------------------ 删（调用方的"让她忘掉"）
    def forget(
        self,
        node_ids: Sequence[int],
        *,
        now: float,
        reason: str = "user_request",
        cascade_derived: bool = True,
    ) -> dict:
        """**硬删除**：节点 + 边 + 标签 + 出处原文一起抹掉，只留**墓碑**（id / 类型 / 时间 / 原因，**不留内容**）。

        这是一次**显式写入**，不是检索的副作用 ⇒ `retrieve()` 仍是纯函数（§14.3 / 硬规矩 3）。

        派生节点（`derived_from` 的 dst：反思 / 偏好归纳）里可能带着被删事件的内容：
          * `cascade_derived=True`：**所有来源都已被删**的派生节点一并删除（它只剩被删内容的转述）；
          * 仍有别的存活来源的派生节点**不删**，但会列在返回值的 `tainted` 里 —— 它的摘要
            可能还转述着被删内容，**由调用方决定**是否重写/删掉（这里不替你猜）。
        返回 `{"deleted": [...], "cascaded": [...], "tainted": [...],
        "trace_deleted": n}`。

        轨迹（辅网）正常路径 append-only；这里是**唯一**的擦除例外：与被删节点
        相关的轨迹行会一起清掉，避免"节点删了、提案原文还在轨迹里"。
        """

        base = {int(i) for i in node_ids}
        existing = self._existing_ids(base)
        deleted = sorted(existing)
        cascaded: list[int] = []
        tainted: list[int] = []

        if existing:
            if cascade_derived:
                cascaded, tainted = self._derived_fallout(existing)
            else:
                _, tainted = self._derived_fallout(existing)
            kill = existing | set(cascaded)
            kill_nodes = self.nodes_by_ids(sorted(kill))
            texts = []
            for n in kill_nodes:
                texts.extend([n.summary, n.source_quote, n.scene or "", n.source_turn])
            keys = [t for t in texts if t and len(t) >= 4]
            safe_reason = str(reason)
            if looks_sensitive(safe_reason) or any(k in safe_reason for k in keys):
                safe_reason = "redacted"
            kinds = dict(self.conn.execute(
                f"SELECT id, kind FROM nodes WHERE id IN ({self._qs(kill)})", sorted(kill)).fetchall())
            ids = sorted(kill)
            q = self._qs(ids)
            self.conn.execute(f"DELETE FROM edges WHERE src IN ({q}) OR dst IN ({q})", ids + ids)
            self.conn.execute(f"DELETE FROM tags WHERE node_id IN ({q})", ids)
            self.conn.execute(f"DELETE FROM nodes WHERE id IN ({q})", ids)
            self.conn.executemany(
                "INSERT INTO tombstones (node_id, kind, deleted_at, reason) VALUES (?,?,?,?)",
                [(i, kinds.get(i, ""), float(now), safe_reason) for i in ids],
            )
            self.conn.commit()
            edge_quotes = self._scrub_edge_quotes(keys)
            trace_deleted = self.purge_trace_for_nodes(ids, texts=texts)
        else:
            edge_quotes = 0
            trace_deleted = 0
        return {"deleted": deleted, "cascaded": sorted(cascaded), "tainted": sorted(tainted),
                "edge_quotes_scrubbed": edge_quotes, "trace_deleted": trace_deleted}

    def forget_by_tags(self, tags: Sequence[str], *, now: float, reason: str = "user_request",
                       cascade_derived: bool = True) -> dict:
        """删掉带任一标签的全部节点（"把我说过的关于 X 的都忘掉"）。"""

        ids = self.node_ids_with_any_tag(tuple(tags))
        return self.forget(ids, now=now, reason=reason, cascade_derived=cascade_derived)

    def scrub(self) -> None:
        """文件库：整理文件，让被删内容不残留在空闲页里（`VACUUM`）。内存库无需。"""

        if self.path != ":memory:":
            self.conn.commit()
            self.conn.execute("VACUUM")

    def tombstones(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT node_id, kind, deleted_at, reason FROM tombstones ORDER BY seq").fetchall()
        return [{"node_id": r[0], "kind": r[1], "deleted_at": r[2], "reason": r[3]} for r in rows]

    # ------------------------------------------------------------ 自我轨迹（辅网 v1）
    # §22.4-②：正常路径**写完冻结** —— 只有 append 与读；为满足被遗忘权，
    # 另设显式 `delete_trace` / `purge_trace_for_nodes` / `redact_trace_payload`。
    # 检索路径永远不读这张表（§22.4-①）。
    def append_trace(self, *, ts: float, kind: str, payload: dict,
                     reject_sensitive: bool = True) -> int:
        """追加一条轨迹。返回 `seq`（单调递增）。

        `reject_sensitive=False` 只用于导入旧版轨迹；新写入不要关闭闸门。
        """

        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if reject_sensitive:
            reject_if_sensitive(body, label="轨迹写入")
        cur = self.conn.execute(
            "INSERT INTO self_trace (ts, kind, payload) VALUES (?,?,?)",
            (float(ts), str(kind), body),
        )
        self.conn.commit()
        return _lastrowid(cur)

    def delete_trace(self, seqs: Sequence[int]) -> int:
        """**显式擦除**轨迹行；只给 `forget()` / 被遗忘权路径使用。"""

        ids = sorted({int(s) for s in seqs})
        if not ids:
            return 0
        q = self._qs(ids)
        cur = self.conn.execute(f"DELETE FROM self_trace WHERE seq IN ({q})", ids)
        self.conn.commit()
        return int(cur.rowcount)

    def purge_trace_for_nodes(self, node_ids: Sequence[int], *,
                              texts: Sequence[str] = ()) -> int:
        """删掉引用了这些节点 id（或包含这些原文）的轨迹行。

        - `evidence` / `produced` / `rejected` / `activation` / `node_id` 等 int 字段；
        - 旧版轨迹里可能残留的 `summary` / `source_quote` / `note` 原文；
        - 返回删除的行数。
        """

        ids = {int(i) for i in node_ids}
        keys = [str(t) for t in texts if t and len(str(t)) >= 4]
        if not ids and not keys:
            return 0
        doomed: list[int] = []
        for row in self.trace_rows():
            payload = row["payload"]
            if ids and any(i in ids for i in _iter_node_ids(payload)):
                doomed.append(int(row["seq"]))
                continue
            if keys and any(
                isinstance(s, str) and any(k in s for k in keys)
                for s in _iter_strings(payload)
            ):
                doomed.append(int(row["seq"]))
        return self.delete_trace(doomed)

    def _scrub_edge_quotes(self, texts: Sequence[str]) -> int:
        """把幸存边里仍然包含被删原文的 `evidence_quote` 清空。"""

        keys = [str(t) for t in texts if t and len(str(t)) >= 4]
        if not keys:
            return 0
        rows = self.conn.execute("SELECT id, evidence_quote FROM edges").fetchall()
        hits = [int(r[0]) for r in rows
                if r[1] and any(k in r[1] for k in keys)]
        if hits:
            q = self._qs(hits)
            self.conn.execute(f"UPDATE edges SET evidence_quote='' WHERE id IN ({q})", hits)
            self.conn.commit()
        return len(hits)

    def redact_trace_payload(self, seq: int, *,
                             drop: Sequence[str] = ("summary", "source_quote", "note")) -> bool:
        """把一条旧轨迹里的自由文本字段抹掉（保留指纹和结构字段）。"""

        row = self.conn.execute(
            "SELECT payload FROM self_trace WHERE seq=?", (int(seq),)).fetchone()
        if row is None:
            return False
        payload = json.loads(row[0])
        changed = False
        for key in drop:
            if key in payload:
                payload.pop(key)
                changed = True
        if changed:
            self.conn.execute(
                "UPDATE self_trace SET payload=? WHERE seq=?",
                (json.dumps(payload, ensure_ascii=False, sort_keys=True), int(seq)),
            )
            self.conn.commit()
            if self.path != ":memory:":
                self.scrub()
        return changed

    def trace_rows(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT seq, ts, kind, payload FROM self_trace ORDER BY seq").fetchall()
        return [
            {"seq": int(r[0]), "ts": float(r[1]), "kind": r[2], "payload": json.loads(r[3])}
            for r in rows
        ]

    def trace_signature(self) -> tuple:
        """全表逐位签名（"写完冻结 / 幂等重放"两条验收的尺子）。"""

        return tuple(
            (r["seq"], round(r["ts"], 6), r["kind"],
             json.dumps(r["payload"], ensure_ascii=False, sort_keys=True))
            for r in self.trace_rows()
        )

    @staticmethod
    def _qs(ids: Iterable[int]) -> str:
        return ",".join("?" for _ in ids)

    def _existing_ids(self, ids: set[int]) -> set[int]:
        if not ids:
            return set()
        ordered = sorted(ids)
        rows = self.conn.execute(
            f"SELECT id FROM nodes WHERE id IN ({self._qs(ordered)})", ordered).fetchall()
        return {int(r[0]) for r in rows}

    def _derived_fallout(self, gone: set[int]) -> tuple[list[int], list[int]]:
        """沿 `derived_from`（src=来源 → dst=派生）找下游：全部来源都被删 ⇒ 级联；否则 ⇒ 受污染。"""

        cascaded: set[int] = set()
        tainted: set[int] = set()
        frontier = set(gone)
        removed = set(gone)
        while frontier:
            ordered = sorted(frontier)
            rows = self.conn.execute(
                f"SELECT DISTINCT dst FROM edges WHERE kind=? AND src IN ({self._qs(ordered)})",
                [EdgeKind.DERIVED_FROM.value, *ordered]).fetchall()
            frontier = set()
            for (dst,) in rows:
                dst = int(dst)
                if dst in removed:
                    continue
                srcs = {int(r[0]) for r in self.conn.execute(
                    "SELECT src FROM edges WHERE kind=? AND dst=?",
                    (EdgeKind.DERIVED_FROM.value, dst)).fetchall()}
                if srcs <= removed:
                    cascaded.add(dst)
                    tainted.discard(dst)
                    removed.add(dst)
                    frontier.add(dst)
                else:
                    tainted.add(dst)
        return sorted(cascaded), sorted(tainted - cascaded)

    def touch_access(self, node_ids: Sequence[int], now: float) -> int:
        """命中：`access_count += 1`、`last_access = now`（§14.3 的活量表）。"""

        ids = sorted({int(i) for i in node_ids})
        if not ids:
            return 0
        q = ",".join("?" for _ in ids)
        self.conn.execute(
            f"UPDATE nodes SET access_count = access_count + 1, last_access = ? WHERE id IN ({q})",
            [float(now), *ids],
        )
        self.conn.commit()
        return len(ids)

    def add_s_use(self, deltas: Sequence[tuple[int, float]]) -> int:
        """§27：复述累积的**使用强度** `s_use += Δ`（活的量，与 `access_count` 同类）。

        ⚠️ 它**不碰** `s0` / `obs` / `scene` —— 冻结规则一个字都不破。
        `deltas` 由调用方（`lifecycle.commit_access`）算好，因为增量要读节点当前的 `s0`。
        """

        rows = [(int(i), float(d)) for i, d in deltas if float(d) > 0.0]
        if not rows:
            return 0
        self.conn.executemany("UPDATE nodes SET s_use = s_use + ? WHERE id = ?",
                              [(d, i) for i, d in rows])
        self.conn.commit()
        return len(rows)

    def strengthen_edges(self, node_ids: Sequence[int], strength: float) -> int:
        """§14.5 的 Hebbian：本次结果集**内部**的边 +strength（只动结果集内的边）。"""

        ids = sorted({int(i) for i in node_ids})
        if len(ids) < 2:
            return 0
        q = ",".join("?" for _ in ids)
        cur = self.conn.execute(
            f"""UPDATE edges SET weight = weight + ?
                WHERE src IN ({q}) AND dst IN ({q})""",
            [float(strength), *ids, *ids],
        )
        self.conn.commit()
        return int(cur.rowcount)

    def decay_edges(self, factor: float) -> int:
        """§14.5：**边也必须有衰减**，否则跑几千轮之后所有边饱和、图退化成全连接。"""

        cur = self.conn.execute("UPDATE edges SET weight = weight * ?", (float(factor),))
        self.conn.commit()
        return int(cur.rowcount)

    # ------------------------------------------------------------ 读
    def nodes(self) -> list[Node]:
        rows = self.conn.execute(
            """SELECT id, kind, summary, scene, tags, obs, modulators, created_at,
                      base_persist, s0, access_count, last_access, s_use, source_quote, source_turn, obs_source
               FROM nodes ORDER BY id"""
        ).fetchall()
        return [
            Node(
                id=int(r[0]),
                kind=NodeKind(r[1]),
                summary=r[2],
                scene=r[3],
                tags=tuple(json.loads(r[4])),
                obs=_obs_from_json(r[5]),
                modulators=_mods_from_json(r[6]),
                created_at=float(r[7]),
                base_persist=float(r[8]),
                s0=float(r[9]),
                access_count=int(r[10]),
                last_access=float(r[11]),
                s_use=float(r[12]),
                source_quote=r[13],
                source_turn=r[14],
                obs_source=r[15],
            )
            for r in rows
        ]

    def nodes_by_ids(self, ids: Sequence[int]) -> list[Node]:
        """只取指定 id 的节点（建链时用，避免每次全表扫）。"""

        wanted = sorted({int(i) for i in ids})
        if not wanted:
            return []
        q = ",".join("?" for _ in wanted)
        rows = self.conn.execute(
            f"""SELECT id, kind, summary, scene, tags, obs, modulators, created_at,
                       base_persist, s0, access_count, last_access, s_use, source_quote, source_turn, obs_source
                FROM nodes WHERE id IN ({q}) ORDER BY id""",
            wanted,
        ).fetchall()
        return [
            Node(
                id=int(r[0]),
                kind=NodeKind(r[1]),
                summary=r[2],
                scene=r[3],
                tags=tuple(json.loads(r[4])),
                obs=_obs_from_json(r[5]),
                modulators=_mods_from_json(r[6]),
                created_at=float(r[7]),
                base_persist=float(r[8]),
                s0=float(r[9]),
                access_count=int(r[10]),
                last_access=float(r[11]),
                s_use=float(r[12]),
                source_quote=r[13],
                source_turn=r[14],
                obs_source=r[15],
            )
            for r in rows
        ]

    def node_ids_with_any_tag(self, tags: Sequence[str]) -> list[int]:
        """哪些节点带了这些标签里的任意一个（§3.1 的 `tags` 表就是干这个的）。"""

        wanted = sorted({t for t in tags if t})
        if not wanted:
            return []
        q = ",".join("?" for _ in wanted)
        rows = self.conn.execute(
            f"SELECT DISTINCT node_id FROM tags WHERE tag IN ({q}) ORDER BY node_id", wanted
        ).fetchall()
        return [int(r[0]) for r in rows]

    def edges(self) -> list[Edge]:
        rows = self.conn.execute(
            "SELECT src, dst, kind, weight, decay_rate, evidence_quote FROM edges ORDER BY src, dst, kind"
        ).fetchall()
        return [
            Edge(
                src=int(r[0]),
                dst=int(r[1]),
                kind=EdgeKind(r[2]),
                weight=float(r[3]),
                decay_rate=float(r[4]),
                evidence_quote=r[5],
            )
            for r in rows
        ]

    def snapshot(self) -> GraphSnapshot:
        return snapshot_from_rows(self.nodes(), self.edges())

    def close(self) -> None:
        self.conn.close()


def snapshot_from_rows(nodes: Iterable[Node], edges: Iterable[Edge]) -> GraphSnapshot:
    """把行变成**只读快照**，并把邻接表按 `(weight↓, dst↑)` 定序。

    定序是"可复现"的物理保证（§16.5）：同快照 + 同参数 ⇒ 逐位相同。
    """

    node_map = {n.id: n for n in nodes}
    edge_tuple = tuple(
        e for e in edges if e.src in node_map and e.dst in node_map
    )
    adj: dict[int, list[Edge]] = {}
    for e in edge_tuple:
        adj.setdefault(e.src, []).append(e)
    frozen_adj = {
        k: tuple(sorted(v, key=lambda e: (-e.weight, e.dst, e.kind.value)))
        for k, v in adj.items()
    }
    return GraphSnapshot(nodes=node_map, edges=edge_tuple, adj=frozen_adj)


def inspect_snapshot(path: str | Path) -> dict:
    """只读检查快照：完整性 + 各表计数（不触发 MemoryStore 迁移）。"""

    snapshot_path = Path(path).expanduser().resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"快照不存在：{snapshot_path}")
    conn = sqlite3.connect(str(snapshot_path))
    try:
        conn.execute("PRAGMA query_only = ON")
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            return {"integrity": integrity, "counts": None}
        counts = {
            "nodes": int(conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]),
            "edges": int(conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]),
            "tags": int(conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]),
            "tombstones": int(conn.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0]),
            "self_trace": int(conn.execute("SELECT COUNT(*) FROM self_trace").fetchone()[0]),
        }
        return {"integrity": integrity, "counts": counts}
    finally:
        conn.close()


def export_database(source: str | Path, dest: str | Path, *,
                    overwrite: bool = False) -> dict:
    """按文件路径导出整库快照；**不打开 MemoryStore，因此不会触发表结构迁移**。"""

    source_path = Path(source).expanduser().resolve()
    dest_path = Path(dest).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"源数据库不存在：{source_path}")
    if source_path == dest_path:
        raise ValueError("导出目标不能是源数据库本身")
    with closing(sqlite3.connect(str(source_path))) as src:
        src.execute("PRAGMA query_only = ON")
        _atomic_backup(src, dest_path, overwrite=overwrite)
    info = inspect_snapshot(dest_path)
    if info["integrity"] != "ok":
        raise RuntimeError(f"导出快照未通过完整性检查：{info['integrity']}")
    return {"path": str(dest_path), "counts": info["counts"], "integrity": "ok"}


def restore_snapshot(snapshot: str | Path, dest: str | Path, *,
                     overwrite: bool = False) -> dict:
    """把快照恢复到**新数据库**；默认拒绝覆盖已有目标。"""

    snapshot_path = Path(snapshot).expanduser().resolve()
    dest_path = Path(dest).expanduser().resolve()
    if snapshot_path == dest_path:
        raise ValueError("恢复目标不能是快照本身")
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"恢复目标已存在：{dest_path}")
    info = inspect_snapshot(snapshot_path)
    if info["integrity"] != "ok":
        raise RuntimeError(f"快照未通过完整性检查：{info['integrity']}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + f".{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    tmp_path.touch(mode=0o600, exist_ok=True)
    _restrict_file(tmp_path)
    try:
        with closing(sqlite3.connect(str(snapshot_path))) as src:
            src.execute("PRAGMA query_only = ON")
            with closing(sqlite3.connect(tmp_path)) as dst:
                src.backup(dst)
                integrity = str(dst.execute("PRAGMA integrity_check").fetchone()[0])
                if integrity != "ok":
                    raise RuntimeError(f"恢复后未通过完整性检查：{integrity}")
                dst.commit()
        os.replace(tmp_path, dest_path)
        _restrict_file(dest_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return {"path": str(dest_path), "counts": info["counts"], "integrity": "ok"}
