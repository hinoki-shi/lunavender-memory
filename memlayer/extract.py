#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提取器（三层接口：L0 规则 / L1 外部提取 / L2 校准）。

**v1 只有 L0**：0 次 LLM 调用、纯规则、可复现。
L1 留给调用方以适配器形式接入，本包只定义 `Extractor` 接口；
换提取器时，检索与存储一行都不用改。

⚠️ 本模块**不 import 任何网络/模型客户端** —— 依赖留在调用方，别灌进记忆层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .types import Modulators, Observation


@dataclass(frozen=True)
class ExtractedEvent:
    """提取器的输出：能直接喂给 `ingest.remember()`。"""

    summary: str
    obs: Observation
    scene: str | None = None
    tags: tuple[str, ...] = ()
    modulators: Modulators = field(default_factory=Modulators)
    persist_key: str = "event"
    source_quote: str = ""


class Extractor(Protocol):
    """提取器接口。L0 规则版 / L1 模型版 / L2 校准版都实现它。"""

    def extract(self, raw: Mapping[str, object]) -> ExtractedEvent:  # pragma: no cover
        ...


# ---------------------------------------------------------------- L0：规则版

# ⚠️ 这几张表是**临时占位**，不是标定结果；只用于跑通链路。
#    正式使用前应由调用方用自己的数据校准。
_CHANNEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "happy": ("开心", "高兴", "爽", "太好了", "喜欢", "fries", "好吃"),
    "angry": ("气", "烦", "讨厌", "生气", "过分"),
    "shame": ("丢脸", "尴尬", "出丑", "羞"),
    # ⚠️ 早期 修：`anticipation` 自 v4 起**不在 CHANNELS 里**（降级成派生值），
    #    继续往里写等于"关键词命中全作废、hope/worry 恒为 0"。
    #    现在拆成两条独立通道的关键词（盼望 / 担心，可以同时命中）。
    "hope": ("期待", "盼", "想去", "希望"),
    "worry": ("担心", "怕", "焦虑", "紧张", "不敢", "万一"),
    "arousal": ("激动", "惊喜", "突然", "！！", "！"),
}
_MODULATOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "first_time": ("第一次", "头一次", "首次"),
    "public": ("当众", "大家", "所有人", "被看见"),
    "self_relevant": ("我", "我自己"),
    "sensory": ("香", "脆", "热", "味"),
}
# 「能不能沉淀成对他的长期了解」的**规则版**判据（L0，给 L1 当空对照用）
_PREFERENCE_KEYWORDS: tuple[str, ...] = ("喜欢", "讨厌", "最爱", "从来", "每次都", "总是", "习惯了", "受不了")


class RuleExtractor:
    """L0 规则版（§12.2 的 L0）：查表 + 关键词，**不调模型**。

    它不追求准确 —— 它的职责是"让整条链路能端到端跑起来"，
    以及给 L1 提供一个**空对照**（"模型比查表强多少"，这个差值才有意义）。
    """

    def __init__(self, *, scene: str | None = None, per_hit: float = 0.35) -> None:
        self.scene = scene
        self.per_hit = per_hit

    def extract(self, raw: Mapping[str, object]) -> ExtractedEvent:
        text = str(raw.get("text", ""))
        summary = str(raw.get("summary", text[:60]))
        channels: dict[str, float] = {}
        for channel, words in _CHANNEL_KEYWORDS.items():
            hits = sum(1 for w in words if w in text)
            channels[channel] = min(1.0, self.per_hit * hits)
        mods = Modulators(
            **{k: any(w in text for w in words) for k, words in _MODULATOR_KEYWORDS.items()}
        )
        importance = min(1.0, 0.2 + 0.2 * sum(1 for w in _CHANNEL_KEYWORDS["arousal"] if w in text))
        raw_tags = raw.get("tags", ())
        tags = (tuple(str(t) for t in raw_tags)
                if isinstance(raw_tags, (list, tuple, set, frozenset)) else ())
        return ExtractedEvent(
            summary=summary,
            obs=Observation(
                happy=channels["happy"],
                angry=channels["angry"],
                shame=channels["shame"],
                hope=channels["hope"],
                worry=channels["worry"],
                anticipation=(channels["hope"] - channels["worry"]),  # 派生值，只为兼容
                arousal=channels["arousal"],
                importance=importance,
            ),
            scene=str(raw.get("scene", self.scene)) if raw.get("scene", self.scene) else None,
            tags=tags,
            modulators=mods,
            persist_key=str(
                raw.get("persist_key")
                or ("preference" if any(w in text for w in _PREFERENCE_KEYWORDS) else "event")
            ),
            source_quote=text[:120],
        )


def sanity_check(pairs: Sequence[tuple[float, float]] = ()) -> dict:
    """给 L1 预留的**校准钩子**（§12.3 的 L2）。

    做法（等仪器好了再实现）：固定锚点样例 + 出处校验 + 秩相关（**不用绝对误差**）。
    现在返回空结构，避免上层以为"校准已经做了"。
    """

    return {"status": "not_implemented", "n_anchors": len(pairs), "note": "等 §12.4 的仪器"}
