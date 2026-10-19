# -*- coding: utf-8 -*-
"""轻量查询改写（L1）：把常见情绪/时间转述扩成记忆里更可能出现的词。

这是可选层，不改变默认检索；不是分词器，也不是向量召回。
只处理少量通用短语，避免把整句意图硬编码成某一题的答案。
"""

from __future__ import annotations

from .lexicon import tokenize

# 短语级替写：保留原词时也可以把 canonical 词加入查询。
PARAPHRASES: tuple[tuple[str, str], ...] = (
    ("心情很差", "难过"),
    ("心情不好", "难过"),
    ("不开心", "难过"),
    ("很难过", "难过"),
    ("太晚", "半夜"),
    ("很晚", "半夜"),
)


def expand_text(text: str) -> str:
    """把常见转述替换成 canonical 文本；不删除原文的其它字符。"""

    out = str(text or "")
    for old, new in PARAPHRASES:
        out = out.replace(old, new)
    return out


def expand_terms(text: str, *, keep_original: bool = True) -> tuple[str, ...]:
    """返回扩展后的字元；默认把原文词元和改写词元合并、去重。"""

    original = tokenize(text)
    expanded = tokenize(expand_text(text))
    if not keep_original:
        return tuple(dict.fromkeys(expanded))
    return tuple(dict.fromkeys((*original, *expanded)))
