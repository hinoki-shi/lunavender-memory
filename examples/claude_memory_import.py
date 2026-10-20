"""Import a Claude data export into a separate local memlayer DB.

The source archive is never modified. Conversation text is summarized at the
conversation level; message bodies are not copied into the memory graph.
Emails / phone numbers / 18-digit IDs are redacted before insertion.

This is an archive/evaluation tool, not a training script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Use the main project's memory layer, which has the entity-tag extractor.
MEM_LAYER = Path(r"D:\Graduation\code\memory_layer")
if str(MEM_LAYER) not in sys.path:
    sys.path.insert(0, str(MEM_LAYER))

from memlayer import MemoryStore, Observation, remember  # noqa: E402
from memlayer.extract import extract_tags  # noqa: E402
from memlayer.types import NodeKind  # noqa: E402

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID18_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def sanitize(text: str) -> str:
    t = str(text or "")
    t = EMAIL_RE.sub("[邮箱已脱敏]", t)
    t = PHONE_RE.sub("[手机号已脱敏]", t)
    t = ID18_RE.sub("[证件号已脱敏]", t)
    return t


def parse_time(value) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def day_of(value) -> float:
    d = parse_time(value)
    return d.timestamp() / 86400.0 if d else 0.0


def scene_of(value) -> str:
    d = parse_time(value)
    return f"{d.month}月{d.day}日" if d else "Claude档案"


def chunks(text: str, size: int = 800) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in parts:
        if len(p) > size:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(p[i:i + size] for i in range(0, len(p), size))
        elif len(buf) + len(p) + 2 <= size:
            buf = (buf + "\n\n" + p).strip()
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True,
                    help="目录：包含 conversations.json 和 memories/*.json")
    ap.add_argument("--db", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.extract)
    conv_path = root / "conversations.json"
    mem_paths = sorted((root / "memories").glob("*.json"))
    if not conv_path.is_file() or not mem_paths:
        print("FAIL: conversations.json / memories 不存在")
        return 2
    db = Path(args.db)
    if db.exists() and not args.overwrite:
        print("FAIL: DB 已存在；换路径或加 --overwrite")
        return 2
    if db.exists():
        db.unlink()

    store = MemoryStore(db)
    try:
        conversations = json.loads(conv_path.read_text(encoding="utf-8"))
        mem_doc = json.loads(mem_paths[0].read_text(encoding="utf-8"))
        imported = 0
        skipped = 0

        # 32 conversations as episode nodes. Do not copy message bodies.
        for conv in conversations:
            name = sanitize(conv.get("name") or "Claude 对话")[:200]
            summary = sanitize(conv.get("summary") or name)[:1200]
            if not summary.strip():
                skipped += 1
                continue
            try:
                remember(
                    store, summary=summary,
                    obs=Observation(importance=0.6),
                    tags=extract_tags(summary + " " + name),
                    scene=scene_of(conv.get("created_at")),
                    created_at=day_of(conv.get("created_at")),
                    persist_key="event",
                    source_quote=name,
                    obs_source="claude_export",
                )
                imported += 1
            except Exception:
                skipped += 1

        # 8 memory files as semantic nodes.
        for item in mem_doc.get("memory_files") or []:
            content = sanitize(item.get("content") or "")[:1200]
            if not content.strip():
                skipped += 1
                continue
            try:
                remember(
                    store, summary=content,
                    obs=Observation(importance=0.6),
                    tags=extract_tags(content),
                    kind=NodeKind.SEMANTIC,
                    scene=None,
                    created_at=day_of(item.get("updated_at")),
                    persist_key="reflection",
                    source_quote=Path(str(item.get("path") or "memory")).name,
                    obs_source="claude_memory_file",
                )
                imported += 1
            except Exception:
                skipped += 1

        # The long conversations_memory summary: chunk into semantic nodes.
        summary_dates = [parse_time(item.get("updated_at"))
                         for item in (mem_doc.get("memory_files") or [])]
        summary_dates = [d for d in summary_dates if d]
        summary_day = (max(summary_dates).timestamp() / 86400.0
                       if summary_dates else 0.0)
        for i, chunk in enumerate(chunks(sanitize(mem_doc.get("conversations_memory") or ""))):
            try:
                remember(
                    store, summary=chunk,
                    obs=Observation(importance=0.6),
                    tags=extract_tags(chunk),
                    kind=NodeKind.SEMANTIC,
                    scene=None,
                    created_at=summary_day,
                    persist_key="reflection",
                    source_quote="conversations_memory#%d" % i,
                    obs_source="claude_memory_summary",
                )
                imported += 1
            except Exception:
                skipped += 1

        counts = store.counts()
        nodes = store.nodes()
        manifest = {
            "source": str(root),
            "db": str(db),
            "conversations": len(conversations),
            "memory_files": len(mem_doc.get("memory_files") or []),
            "imported_nodes": imported,
            "skipped": skipped,
            "counts": counts,
            "date_min": min((n.created_at for n in nodes), default=None),
            "date_max": max((n.created_at for n in nodes), default=None),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [conv_path, *mem_paths]
            },
        }
        out = db.with_suffix(".manifest.json")
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(json.dumps({k: manifest[k] for k in (
            "conversations", "memory_files", "imported_nodes", "skipped",
            "counts", "date_min", "date_max")}, ensure_ascii=False, indent=2))
        print("manifest:", out)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
