# -*- coding: utf-8 -*-
"""近期工作记忆缓冲单测。"""

from __future__ import annotations

import unittest

from memlayer.recent_buffer import RecentBuffer, estimate_tokens


class TestRecentBuffer(unittest.TestCase):
    def test_threshold_and_order(self) -> None:
        buf = RecentBuffer(token_counter=lambda u, a: len(u) + len(a))
        self.assertTrue(buf.add_turn(session_id="s", turn_id="1", ts=1.0,
                                    user_text="1234", assistant_text="5678"))
        self.assertFalse(buf.should_flush(high_tokens=20))
        buf.add_turn(session_id="s", turn_id="2", ts=2.0,
                     user_text="abcd", assistant_text="efgh")
        self.assertTrue(buf.should_flush(high_tokens=16))
        out = buf.drain()
        self.assertEqual([t["turn_id"] for t in out], ["1", "2"])
        self.assertEqual(buf.turn_count, 0)

    def test_duplicate_turn_is_ignored(self) -> None:
        buf = RecentBuffer(token_counter=lambda u, a: 10)
        self.assertTrue(buf.add_turn(session_id="s", turn_id="1", ts=1.0,
                                    user_text="a", assistant_text="b"))
        self.assertFalse(buf.add_turn(session_id="s", turn_id="1", ts=2.0,
                                     user_text="duplicate", assistant_text=""))
        self.assertEqual(buf.turn_count, 1)
        self.assertEqual(buf.total_tokens, 10)

    def test_flushed_turn_cannot_be_reintroduced(self) -> None:
        buf = RecentBuffer(token_counter=lambda u, a: 10)
        buf.add_turn(session_id="s", turn_id="1", ts=1.0,
                     user_text="u", assistant_text="a")
        buf.drain()
        self.assertFalse(buf.add_turn(session_id="s", turn_id="1", ts=2.0,
                                     user_text="u", assistant_text="a"))

    def test_tail_retention_keeps_complete_turns(self) -> None:
        buf = RecentBuffer(token_counter=lambda u, a: 10)
        for i in range(3):
            buf.add_turn(session_id="s", turn_id=str(i), ts=float(i),
                         user_text="u", assistant_text="a")
        flushed = buf.drain(retain_tail_tokens=12)
        self.assertEqual([t["turn_id"] for t in flushed], ["0", "1"])
        self.assertEqual([t["turn_id"] for t in buf], ["2"])
        self.assertEqual(buf.total_tokens, 10)

    def test_snapshot_roundtrip(self) -> None:
        buf = RecentBuffer(token_counter=lambda u, a: 3)
        buf.add_turn(session_id="s", turn_id="1", ts=1.0,
                     user_text="我", assistant_text="在")
        restored = RecentBuffer.from_dict(buf.to_dict(), token_counter=lambda u, a: 3)
        self.assertEqual(restored.to_dict(), buf.to_dict())
        self.assertEqual(restored.total_tokens, 3)

    def test_token_estimate_has_cjk_and_latin_paths(self) -> None:
        self.assertGreaterEqual(estimate_tokens("你好世界"), 4)
        self.assertGreater(estimate_tokens("hello world"), 0)
        self.assertEqual(estimate_tokens(""), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
