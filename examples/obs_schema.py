# -*- coding: utf-8 -*-
"""事件级观测量 schema（9 个字段，0–4 档）。

这是可选的 9 通道观测 schema。标注、推理 question 和把 0–4 分转成
`Observation` 都应走这里，避免不同环节的口径静默错位。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping

MEM_LAYER = Path(__file__).resolve().parents[1]
if str(MEM_LAYER) not in sys.path:
    sys.path.insert(0, str(MEM_LAYER))

from memlayer.types import Modulators, Observation  # noqa: E402

FIELDS: tuple[str, ...] = (
    "happy", "angry", "shame", "hope", "worry",
    "anxiety", "arousal", "importance", "unfinished",
)

_GENERIC = ("完全没有", "很弱", "中等", "比较强", "非常强")
_IMPORTANCE = ("无关紧要", "轻微", "中等", "比较重要", "非常重要")
_UNFINISHED = ("已经了结", "基本了结", "有点挂念", "经常想起", "一直悬着/反复想")

QUESTIONS: dict[str, dict] = {
    "happy": {
        "instructions": "这段经历里，**开心/愉悦**的成分有多强？",
        "criteria": _GENERIC,
    },
    "angry": {
        "instructions": "这段经历里，**生气/恼火**的成分有多强？",
        "criteria": _GENERIC,
    },
    "shame": {
        "instructions": "这段经历里，**丢脸/尴尬/羞耻**的成分有多强？",
        "criteria": _GENERIC,
    },
    "hope": {
        "instructions": "这段经历里，对**之后的好事**的期待有多强？",
        "criteria": _GENERIC,
    },
    "worry": {
        "instructions": "这段经历里，对**之后的坏事**的担心有多强？",
        "criteria": _GENERIC,
    },
    "anxiety": {
        "instructions": "他**内心不安/悬着**的程度有多强？（0=很平静，4=一直悬着/发慌；只看情绪波动，不把反复刷消息当不平静）",
        "criteria": ("很平静", "比较平静", "中等", "有点悬着", "一直悬着/发慌"),
    },
    "arousal": {
        "instructions": "他当时的**激动程度**有多强？（心跳快 / 坐不住 / 脑子转得快 / 手抖；好坏方向都算）",
        "criteria": _GENERIC,
    },
    "importance": {
        "instructions": "这件事对**他自己**的生活有多重要？（一年后回头看是否值得记住 / 以后还用不用得上）",
        "criteria": _IMPORTANCE,
    },
    "unfinished": {
        "instructions": "这件事到现在**有结果了吗 / 他还会不会反复想起来**？",
        "criteria": _UNFINISHED,
    },
}


def question_specs() -> dict[str, dict]:
    """返回可直接交给兼容模型的 question 字典。"""
    return {
        field: {
            "type": "score",
            "instructions": spec["instructions"],
            "criteria": list(spec["criteria"]),
        }
        for field, spec in QUESTIONS.items()
    }


def score_to_distribution(value: float, *, sigma: float = 0.45,
                          n: int = 5) -> dict[str, float]:
    """把 0–4 的连续标签变成 5 档软分布。

    软目标保留 0.5 档信息；如果直接用 nearest one-hot，会把调用方标的
    “中间偏一点”压成整数，丢掉量表里最贵的细粒度。
    """
    v = max(0.0, min(float(n - 1), float(value)))
    s = max(1e-6, float(sigma))
    raw = [math.exp(-0.5 * ((i - v) / s) ** 2) for i in range(n)]
    total = sum(raw)
    return {str(i): round(w / total, 8) for i, w in enumerate(raw)}


def make_gold(field: str, value: float) -> dict:
    v = max(0.0, min(4.0, float(value)))
    probs = score_to_distribution(v)
    top = max(probs, key=lambda k: probs[k])
    return {
        "type": "score",
        "score": v,
        "probabilities": probs,
        "label": top,
        "legend": {str(i): text for i, text in enumerate(QUESTIONS[field]["criteria"])},
        "confidence": None,
    }


def values_to_observation(values: Mapping[str, float],
                          *, toggles: Mapping[str, bool] | None = None
                          ) -> tuple[Observation, Modulators]:
    """把 0–4 的评分转成记忆层观测量（0–1）。"""
    def unit(name: str) -> float:
        return max(0.0, min(1.0, float(values.get(name, 0.0)) / 4.0))

    hope, worry = unit("hope"), unit("worry")
    obs = Observation(
        happy=unit("happy"),
        angry=unit("angry"),
        shame=unit("shame"),
        hope=hope,
        worry=worry,
        anticipation=hope - worry,
        anxiety=unit("anxiety"),
        arousal=unit("arousal"),
        importance=unit("importance"),
        unfinished=unit("unfinished"),
    )
    t = toggles or {}
    mods = Modulators(
        first_time=bool(t.get("first_time", False)),
        public=bool(t.get("public", False)),
        self_relevant=bool(t.get("self_relevant", False)),
        sensory=bool(t.get("sensory", False)),
    )
    return obs, mods
