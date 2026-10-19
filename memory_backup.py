#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆库可验证备份：SQLite 快照 + JSON 清单。

设计底线：
* 导出用 SQLite backup API，包含节点、边、标签、墓碑、自我轨迹；
* 清单记录 schema 版本、各表计数、文件大小和 SHA-256；
* 恢复只写新库，默认拒绝覆盖；先验清单，再 backup。

备份文件是**明文**，里面有真实记忆；请放到 IKKP200 等离线介质，不要上传。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from memlayer.store import (
    SCHEMA_VERSION,
    export_database,
    inspect_snapshot,
    restore_snapshot,
)

FORMAT = "lunavender-memory-snapshot/v1"
MANIFEST_SUFFIX = ".manifest.json"


class BackupVerificationError(ValueError):
    """清单、哈希或数据库内容对不上。"""


def manifest_path(snapshot: str | Path) -> Path:
    return Path(str(snapshot) + MANIFEST_SUFFIX).expanduser().resolve()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        tmp.touch(mode=0o600, exist_ok=True)
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if tmp.exists():
            tmp.unlink()


def export_memory(db_path: str | Path, snapshot_path: str | Path, *,
                  overwrite: bool = False) -> dict:
    """把文件数据库导出为快照，并写同名 `.manifest.json`。"""

    source = Path(db_path).expanduser().resolve()
    snapshot = Path(snapshot_path).expanduser().resolve()
    manifest_file = manifest_path(snapshot)
    if manifest_file.exists() and not overwrite:
        raise FileExistsError(f"清单已存在：{manifest_file}")

    result = export_database(source, snapshot, overwrite=overwrite)
    info = inspect_snapshot(snapshot)
    if info["integrity"] != "ok" or info["counts"] != result["counts"]:
        raise BackupVerificationError("导出后快照自检失败")

    manifest = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_name": source.name,
        "snapshot_name": snapshot.name,
        "bytes": snapshot.stat().st_size,
        "sha256": sha256_file(snapshot),
        "integrity": info["integrity"],
        "counts": info["counts"],
    }
    _write_json_atomic(manifest_file, manifest)
    return {**manifest, "snapshot": str(snapshot), "manifest": str(manifest_file)}


def load_manifest(snapshot: str | Path,
                  manifest: Optional[str | Path] = None) -> dict:
    path = Path(manifest).expanduser().resolve() if manifest else manifest_path(snapshot)
    if not path.is_file():
        raise FileNotFoundError(f"备份清单不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise BackupVerificationError(f"备份清单读不出来：{e}") from e
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise BackupVerificationError("不是本项目的记忆备份格式")
    return data


def verify_memory_export(snapshot: str | Path,
                         manifest: Optional[str | Path] = None) -> dict:
    """校验快照哈希、SQLite 完整性和各表计数。"""

    snap = Path(snapshot).expanduser().resolve()
    if not snap.is_file():
        raise FileNotFoundError(f"快照不存在：{snap}")
    data = load_manifest(snap, manifest)
    actual_bytes = snap.stat().st_size
    if actual_bytes != data.get("bytes"):
        raise BackupVerificationError(
            f"文件大小不符：清单 {data.get('bytes')} / 实际 {actual_bytes}")
    actual_sha = sha256_file(snap)
    if actual_sha != data.get("sha256"):
        raise BackupVerificationError("SHA-256 不符，快照可能损坏或被替换")
    info = inspect_snapshot(snap)
    if info["integrity"] != "ok":
        raise BackupVerificationError(f"SQLite 完整性检查失败：{info['integrity']}")
    if info["counts"] != data.get("counts"):
        raise BackupVerificationError(
            f"表计数不符：清单 {data.get('counts')} / 实际 {info['counts']}")
    return {**data, "snapshot": str(snap), "verified": True}


def restore_memory(snapshot: str | Path, target: str | Path, *,
                   manifest: Optional[str | Path] = None,
                   overwrite: bool = False) -> dict:
    """先验清单，再恢复到新数据库；默认不覆盖任何已有目标。"""

    verified = verify_memory_export(snapshot, manifest)
    result = restore_snapshot(snapshot, target, overwrite=overwrite)
    if result["counts"] != verified["counts"]:
        raise BackupVerificationError(
            f"恢复后计数不符：清单 {verified['counts']} / 实际 {result['counts']}")
    return {**result, "verified": True}
