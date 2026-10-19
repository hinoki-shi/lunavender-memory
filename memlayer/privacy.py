#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""敏感内容拒写 + **产物擦除**（零依赖层）。

背景（代码审查 早期 第 3 条）：旧的 `legacy store` 有 `looks_sensitive()`，
新的 `memlayer/` 没有。她能记住羞耻 / 愤怒 / 不安，就更不该把凭据写进磁盘。

规则与旧实现**同向**（宁可少记，也不要把凭据落盘）；这里复制而不是 import，
因为本包的第一条规矩是"零项目内依赖"。

## ⚠️ 早期 加固（评审自查发现的三个漏洞，全部有测试）

1. **身份证末位是 X 会漏**：原来只有 `\\b\\d{15,19}\\b`，而 `110101199003071XX` 这类
   在 `X` 之前**没有词边界** ⇒ 整条漏掉。现在单独识别 18 位身份证并**算校验位**。
2. **16 位数字被误杀**：微秒时间戳（如 `1728451200123456`）会命中 `\\d{15,19}` ⇒ 正常的记忆写不进去。
   现在对纯数字长串先做三步判断：**Luhn 通过 ⇒ 卡号**；否则**像时间戳（ms/µs/ns 三个量级）⇒ 放过**；
   否则若**附近有"身份证/卡号/账号"之类的语境词 ⇒ 卡号**；都不满足才算可疑长数字。
3. **删除只覆盖了库、没覆盖产物**：`store.forget()` 抹的是 SQLite 里的节点/边/标签/出处，
   但同一段原文**还躺在外部打标缓存里**（实测：拿事件开头 12 个字去搜外部打标缓存，
   命中 5 行）。⇒ 本模块补 `find_text_in_files()` / `scrub_files()`：
   **先只读扫、再逐行擦**（默认留 `.bak`），让"让她忘掉"这条链路能真的走完。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

_SECRET_PATTERNS = (
    r"\bsk-[A-Za-z0-9_-]{16,}",                       # OpenAI 风格
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",                  # GitHub token
    r"(?i)\b(api[_-]?key|apikey|secret|token|password|passwd|pwd)\b\s*[:=：]\s*\S+",
    r"(?i)\bBearer\s+[A-Za-z0-9._-]{20,}",
)
_SECRET_RE = tuple(re.compile(p) for p in _SECRET_PATTERNS)

# ---- 数字类：靠**校验位**判，而不是靠位数（见文件头"加固"第 1、2 条）----
_DIGIT_RUN_RE = re.compile(r"(?<!\d)(\d{15,19})(?!\d)")
_ID18_RE = re.compile(r"(?<![0-9A-Za-z])(\d{17}[\dXx])(?![0-9A-Za-z])")
_CONTEXT_RE = re.compile(r"(身份证|证件号|银行卡|卡号|账号|账户|card|iban|account)", re.I)
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK = "10X98765432"
# 时间戳三个量级的合法区间（早期 ~ 2100-01-01 UTC）；用来**放过**正常日志里的时间戳
_EPOCH_LO = 978307200
_EPOCH_HI = 4102444800
_EPOCH_SCALES = (10 ** 3, 10 ** 6, 10 ** 9)


def _id18_valid(s: str) -> bool:
    """中国居民身份证 18 位的**校验位**校验（GB 11643）。"""

    if len(s) != 18 or not s[:17].isdigit():
        return False
    total = sum(int(d) * w for d, w in zip(s[:17], _ID_WEIGHTS))
    return _ID_CHECK[total % 11] == s[17].upper()


def _luhn_ok(digits: str) -> bool:
    """Luhn 校验（银行卡）。"""

    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _epoch_like(n: int) -> bool:
    return any(_EPOCH_LO * s <= n <= _EPOCH_HI * s for s in _EPOCH_SCALES)


class SensitiveContentError(ValueError):
    """写入被拒：内容疑似凭据。**异常信息里不带原文**，避免凭据又被写进日志。"""


def looks_sensitive(text: str) -> Optional[str]:
    """命中返回**命中类别**（不是命中的原文），干净返回 None。"""

    t = text or ""
    for idx, rx in enumerate(_SECRET_RE):
        if rx.search(t):
            return f"pattern#{idx}"
    # ① 身份证（18 位，末位可能是 X）：**校验位通过**才算（避免误杀 18 位随机数字）
    for m in _ID18_RE.finditer(t):
        if _id18_valid(m.group(1)):
            return "id_card"
    # ② 纯数字长串：Luhn / 时间戳 / 语境词 三步
    for m in _DIGIT_RUN_RE.finditer(t):
        digits = m.group(1)
        if _luhn_ok(digits):
            return "card_luhn"
        window = t[max(0, m.start() - 16): m.end() + 16]
        if _CONTEXT_RE.search(window):
            return "card_context"
        if _epoch_like(int(digits)):
            continue                      # 像时间戳 ⇒ 放过（加固第 2 条）
        return "long_digits"
    return None


# ---------------------------------------------------------------- 产物擦除（加固第 3 条）

def find_text_in_files(paths: Iterable[str | Path], needles: Sequence[str]) -> dict[str, int]:
    """**只读**扫描：在每个文件里数一数，有多少行包含任一 `needles`。

    删除流程先跑它 —— 先让调用方看见"原文还残留在哪几个文件里"，再决定擦不擦。
    返回 `{文件路径: 命中行数}`（只列命中的）。
    """

    keys = [n for n in needles if n]
    out: dict[str, int] = {}
    if not keys:
        return out
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        hits = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(k in line for k in keys):
                hits += 1
        if hits:
            out[str(path)] = hits
    return out


def scrub_files(
    paths: Iterable[str | Path],
    needles: Sequence[str],
    *,
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, dict]:
    """逐行擦除含 `needles` 的行（默认留 `.bak` 再写回）。

    ⚠️ **这是"让她忘掉"链路里唯一动磁盘产物的入口** —— `store.forget()` 只覆盖 SQLite；
    外部打标缓存必须靠这里擦。

    返回 `{文件路径: {"removed": n, "kept": m, "backup": ".bak 路径或 None"}}`。
    """

    keys = [n for n in needles if n]
    report: dict[str, dict] = {}
    if not keys:
        return report
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        kept = [ln for ln in lines if not any(k in ln for k in keys)]
        removed = len(lines) - len(kept)
        if removed == 0:
            report[str(path)] = {"removed": 0, "kept": len(lines), "backup": None}
            continue
        bak = None
        if not dry_run:
            if backup:
                bak_path = path.with_suffix(path.suffix + ".bak")
                bak_path.write_text("".join(lines), encoding="utf-8")
                bak = str(bak_path)
            path.write_text("".join(kept), encoding="utf-8")
        report[str(path)] = {"removed": removed, "kept": len(kept), "backup": bak}
    return report
