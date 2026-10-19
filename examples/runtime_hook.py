#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对话运行时 ↔ 记忆层的示例接缝（规则版写入 + 触发词召回）。

这是一个最小示例：拿到调用方文本后 `remember_turn(...)`；回复前
`recall_context(...)`，非空就拼进 system prompt。调用方可以把这两个函数
接到自己的对话循环里。

原则：
* **零网络、零 LLM 调用**（提取用 L0 规则版；检索是纯函数）；
* 写入过隐私闸：疑似证件号/卡号**拒写**（返回 None，绝不打断对话）；
* 召回只在**回忆触发词**出现时进行，且用**事实档** `params_for_path("fact")`；
* 注入模板防幻觉：注明来源、带日期、附"不要编造列表之外"的提醒。

⚠️ **时间单位约定（早期 修）**：库里一律用「**天**」（与全部实验 / 合成库一致；
`recency` 的 τ=1 天、半衰期 7 天等参数都按天解释）。本模块对外**收 epoch 秒**
（`time.time()` 直传即可），在边界处换算 —— 之前直接落秒会让记忆一天后 recency≈0.003。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DAYS = 86400.0   # 库内时间单位＝天；对外接口＝epoch 秒

from memlayer import (  # noqa: E402
    MemoryStore,
    Node,
    Params,
    Query,
    params_for_path,
    remember,
    retrieve,
    tokenize,
)
from memlayer.extract import RuleExtractor  # noqa: E402
from memlayer.privacy import SensitiveContentError  # noqa: E402
from memlayer.query_expand import expand_terms  # noqa: E402

# 回忆触发词：只有出现这些词才去翻记忆（"省、准、可控"）
RECALL_TRIGGERS: tuple[str, ...] = (
    "上次", "之前", "那天", "那次", "那时候", "当时", "还记得", "记得",
    "以前", "上周", "上个月", "昨天", "前天", "好久", "多久", "后来",
)

# 关键词提取的粗停用字表；生产接真分词/实体时替换
_STOP_CHARS = set(
    "的了在是我你他她它这一有不就都也很把被和跟与或吗呢啊吧哦嘛么次那这"
    "个些没还又再会让给到过太好多多少什怎什么怎样呀啦对从向当只才已"
    "应该能会想说要去看做发生在后前时回上下里外中"
)


def _slot_of(now: float) -> str:
    h = datetime.fromtimestamp(now).hour
    if 5 <= h < 11:
        return "早上"
    if 11 <= h < 14:
        return "中午"
    if 14 <= h < 18:
        return "下午"
    if 18 <= h < 23:
        return "晚上"
    return "深夜"


def runtime_scene(now: float) -> str:
    """运行时的场景键（近似：`「M月D日·时段」`，没有地点源——方案 v1 §2/§6）。"""

    d = datetime.fromtimestamp(now)
    return f"{d.month}月{d.day}日·{_slot_of(now)}"


def should_recall(user_text: str) -> bool:
    t = str(user_text or "")
    return any(w in t for w in RECALL_TRIGGERS)


def extract_query_terms(user_text: str, *, expand: bool = False) -> tuple[str, ...]:
    """查询关键词：分词后去停用字；整句直喂容易被显著度抢种子。"""

    toks = tuple(t for t in tokenize(user_text) if t not in _STOP_CHARS)
    if expand:
        extra = tuple(t for t in expand_terms(user_text) if t not in _STOP_CHARS)
        toks = tuple(dict.fromkeys((*toks, *extra)))
    return toks or tuple(tokenize(user_text))


def remember_turn(store: MemoryStore, *, text: str, now: float,
                  scene: Optional[str] = None,
                  extractor: Optional[RuleExtractor] = None) -> Optional[Node]:
    """把一轮调用方发言写进记忆（P0 的核心调用）。敏感内容/过短 → 返回 None（静默）。

    `now` 收 **epoch 秒**（`time.time()` 直传），内部换算成天入库。
    """

    text = str(text or "").strip()
    if len(text) < 4:
        return None
    scene_key = scene if scene is not None else runtime_scene(now)
    ex = extractor or RuleExtractor(scene=scene_key)
    try:
        ev = ex.extract({"text": text, "summary": text[:60], "scene": scene_key})
        return remember(
            store,
            summary=ev.summary,
            obs=ev.obs,
            scene=ev.scene,
            tags=ev.tags,
            modulators=ev.modulators,
            created_at=float(now) / DAYS,     # ← epoch 秒 → 天（边界换算）
            persist_key=ev.persist_key,
            source_quote=ev.source_quote,
            obs_source="runtime_rule",
        )
    except SensitiveContentError:
        return None


def recall_context(store: MemoryStore, *, user_text: str, now: float, k: int = 4,
                   params: Optional[Params] = None, expand: bool = False) -> Optional[dict]:
    """需要回忆时（触发词命中）返回注入块；否则 None。

    返回：`{"text": 渲染好的注入块, "ids": [...], "terms": [...], "path": "fact"}`
    """

    if not should_recall(user_text):
        return None
    terms = extract_query_terms(user_text, expand=expand)
    if not terms:
        return None
    p = params if params is not None else params_for_path("fact")
    snap = store.snapshot()
    r = retrieve(snap, Query(terms=terms, now=float(now) / DAYS), 0.5, p)
    hits = r.top(k=k)
    if not hits:
        return None
    lines = []
    for nid, _score in hits:
        node = snap.nodes[nid]
        when = datetime.fromtimestamp(node.created_at * DAYS).strftime("%m月%d日") \
            if node.created_at else "早前"
        lines.append(f"{len(lines) + 1}. （{when}）{node.summary}")
    body = "\n".join(lines)
    text = (
        "【以下是你记得的事（来自你们之前的对话记录）】\n"
        f"{body}\n"
        "提醒：只在本轮相关时自然提起；不要编造列表之外的记忆。"
    )
    return {"text": text, "ids": [nid for nid, _ in hits], "terms": list(terms), "path": "fact"}
