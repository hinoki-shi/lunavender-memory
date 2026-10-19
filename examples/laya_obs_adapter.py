#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional adapter: run Laya's typed decision API for observation fields.

This file is an example, not a bundled model. Install `laya` from upstream,
obtain its weights from the official model hub, and fine-tune/evaluate your own
checkpoint for your data. No Laya weights are included in this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent
for p in (str(ROOT), str(EXAMPLES)):
    if p not in sys.path:
        sys.path.insert(0, p)

from obs_schema import FIELDS, question_specs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Optional Laya observation adapter")
    ap.add_argument("--text", required=True)
    ap.add_argument("--scene", default="")
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default="multilingual",
                    help="upstream checkpoint subfolder, e.g. multilingual")
    ap.add_argument("--local-path", default="",
                    help="local fine-tuned checkpoint directory; overrides --model")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    try:
        import laya
    except ImportError:
        print("Laya is not installed. Install it from upstream first: pip install laya")
        return 2

    try:
        if args.local_path:
            agent = laya.load(args.local_path, device=args.device)
        else:
            agent = laya.load(args.model, subfolder=args.subfolder,
                              device=args.device)
        result = agent.predict(
            {"text": args.text, "scene": args.scene},
            question_specs(),
        )
        answers = result.get("answers", result)
        values = {}
        for field in FIELDS:
            item = answers.get(field, {})
            values[field] = round(float(item.get("score", 0.0)), 4)
        print(json.dumps({
            "values": values,
            "note": "Base/checkpoint output only; fine-tune and evaluate on your own data.",
        }, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"Laya adapter failed: {exc!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
