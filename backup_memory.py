#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆快照命令行：export / verify / restore。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from memory_backup import export_memory, restore_memory, verify_memory_export  # noqa: E402


def _default_out(db_path: Path) -> Path:
    root = HERE.parents[1]
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
    return root / "outputs" / "memory" / "backups" / f"memory-{stamp}.sqlite3"


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="记忆库快照导出/校验/恢复")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="导出带 SHA-256 清单的整库快照")
    ex.add_argument("--db", required=True, help="源 SQLite 数据库")
    ex.add_argument("--out", help="输出快照；默认 outputs/memory/backups/memory-时间.sqlite3")
    ex.add_argument("--overwrite", action="store_true")

    ve = sub.add_parser("verify", help="校验快照与清单")
    ve.add_argument("--snapshot", required=True)
    ve.add_argument("--manifest")

    re = sub.add_parser("restore", help="校验后恢复到新数据库")
    re.add_argument("--snapshot", required=True)
    re.add_argument("--out", required=True)
    re.add_argument("--manifest")
    re.add_argument("--overwrite", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "export":
        db = Path(args.db).expanduser().resolve()
        out = Path(args.out).expanduser().resolve() if args.out else _default_out(db)
        result = export_memory(db, out, overwrite=args.overwrite)
    elif args.cmd == "verify":
        result = verify_memory_export(args.snapshot, args.manifest)
    else:
        result = restore_memory(args.snapshot, args.out, manifest=args.manifest,
                                overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
