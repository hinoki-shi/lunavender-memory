# -*- coding: utf-8 -*-
"""近期工作记忆缓冲（写侧，未接主链路）。

近期对话先留在长上下文里；达到 token 高水位、话题切换或会话结束时，再把完整轮次
交给主网。缓冲区只保存原文和追溯键，不是第二套长期记忆。
"""

from __future__ import annotations

import math
from typing import Callable, Iterator, Optional

TokenCounter = Callable[[str, str], int]


def estimate_tokens(user_text: str, assistant_text: str = "") -> int:
    """零依赖 token 粗估：CJK 按 1 字/token，其余按约 4 字符/token。

    仅用于早期阈值探索；正式实验必须注入真实 tokenizer 计数。
    """

    text = str(user_text or "") + str(assistant_text or "")
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff"
              or "\u3040" <= ch <= "\u30ff")
    other = len(text) - cjk
    return int(cjk + math.ceil(other / 4))


class RecentBuffer:
    """按完整 user+assistant 轮次保存近期对话。"""

    def __init__(self, *, token_counter: Optional[TokenCounter] = None) -> None:
        self._count = token_counter or estimate_tokens
        self._turns: list[dict] = []
        self._keys: set[tuple[str, str]] = set()
        self._seen_keys: set[tuple[str, str]] = set()

    def add_turn(self, *, session_id: str, turn_id: str, ts: float,
                 user_text: str, assistant_text: str = "") -> bool:
        """加入一轮；重复的 `(session_id, turn_id)` 返回 False，不重复计数。"""

        key = (str(session_id), str(turn_id))
        if key in self._seen_keys:
            return False
        tokens = max(0, int(self._count(str(user_text or ""), str(assistant_text or ""))))
        self._turns.append({
            "session_id": key[0],
            "turn_id": key[1],
            "ts": float(ts),
            "user": str(user_text or ""),
            "assistant": str(assistant_text or ""),
            "tokens": tokens,
        })
        self._keys.add(key)
        self._seen_keys.add(key)
        return True

    @property
    def total_tokens(self) -> int:
        return sum(int(t["tokens"]) for t in self._turns)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def should_flush(self, high_tokens: int, *, min_turns: int = 1) -> bool:
        return (self.turn_count >= max(1, int(min_turns))
                and self.total_tokens >= max(0, int(high_tokens)))

    def drain(self, *, retain_tail_tokens: int = 0) -> list[dict]:
        """取出可 flush 的完整轮次；保留最后若干个完整轮次作为连续对话尾巴。

        返回值按加入顺序排列；缓冲区清空后，保留尾巴重新计入下次阈值。
        """

        if not self._turns:
            return []
        keep_tokens = max(0, int(retain_tail_tokens))
        keep_from = len(self._turns)
        kept = 0
        for i in range(len(self._turns) - 1, -1, -1):
            t = int(self._turns[i]["tokens"])
            if kept + t > keep_tokens:
                break
            kept += t
            keep_from = i
        out = self._turns[:keep_from]
        tail = self._turns[keep_from:]
        self._turns = tail
        self._keys = {(str(t["session_id"]), str(t["turn_id"])) for t in tail}
        return out

    def to_dict(self) -> dict:
        """可写进 session journal 的快照（不是主网节点）。"""

        return {
            "turns": [dict(t) for t in self._turns],
            "seen_keys": [list(k) for k in sorted(self._seen_keys)],
        }

    @classmethod
    def from_dict(cls, data: dict, *,
                  token_counter: Optional[TokenCounter] = None) -> "RecentBuffer":
        buf = cls(token_counter=token_counter)
        for t in (data or {}).get("turns", []):
            buf.add_turn(
                session_id=str(t["session_id"]),
                turn_id=str(t["turn_id"]),
                ts=float(t["ts"]),
                user_text=str(t.get("user", "")),
                assistant_text=str(t.get("assistant", "")),
            )
        for k in (data or {}).get("seen_keys", []):
            buf._seen_keys.add((str(k[0]), str(k[1])))
        return buf

    def __iter__(self) -> Iterator[dict]:
        return iter(self._turns)
