# -*- coding: utf-8 -*-
"""观测量 schema 与数据行格式的纯逻辑测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
for p in (str(ROOT), str(EXAMPLES)):
    if p not in sys.path:
        sys.path.insert(0, p)

from obs_schema import (  # noqa: E402
    FIELDS,
    make_gold,
    question_specs,
    score_to_distribution,
    values_to_observation,
)


class TestObsSchema(unittest.TestCase):
    def test_nine_fields_and_five_level_questions(self) -> None:
        self.assertEqual(len(FIELDS), 9)
        specs = question_specs()
        self.assertEqual(set(specs), set(FIELDS))
        for field, spec in specs.items():
            self.assertEqual(spec["type"], "score", field)
            self.assertEqual(len(spec["criteria"]), 5, field)

    def test_soft_score_preserves_midpoint_and_sums_to_one(self) -> None:
        probs = score_to_distribution(2.5)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        expected = sum(int(k) * v for k, v in probs.items())
        self.assertAlmostEqual(expected, 2.5, places=2)

    def test_gold_has_five_level_legend(self) -> None:
        gold = make_gold("anxiety", 3.0)
        self.assertEqual(gold["type"], "score")
        self.assertEqual(len(gold["legend"]), 5)
        self.assertAlmostEqual(sum(gold["probabilities"].values()), 1.0, places=6)

    def test_values_to_observation_scales_to_unit(self) -> None:
        obs, mods = values_to_observation({
            "happy": 4, "angry": 0, "shame": 2,
            "hope": 3, "worry": 1, "anxiety": 2,
            "arousal": 4, "importance": 3, "unfinished": 1,
        }, toggles={"public": True, "self_relevant": True})
        self.assertEqual(obs.happy, 1.0)
        self.assertEqual(obs.angry, 0.0)
        self.assertAlmostEqual(obs.anticipation, 0.5)
        self.assertEqual(obs.arousal, 1.0)
        self.assertTrue(mods.public)
        self.assertTrue(mods.self_relevant)


if __name__ == "__main__":
    unittest.main(verbosity=2)
