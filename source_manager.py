from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from source_rank import SourceRank, rank_source


@dataclass(frozen=True)
class InformationSourceProfile:
    source_name: str
    source_level: int
    source_score: int
    is_original: bool
    is_repeat: bool
    information_type: str
    category: str
    description: str
    duplicate_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_level": self.source_level,
            "source_score": self.source_score,
            "is_original": self.is_original,
            "is_repeat": self.is_repeat,
            "information_type": self.information_type,
            "category": self.category,
            "description": self.description,
            "duplicate_key": self.duplicate_key,
        }


INFORMATION_TYPE_KEYWORDS = {
    "公司公告": ("公告", "巨潮", "上交所", "深交所", "北交所", "回购", "业绩预告"),
    "政策信息": ("国务院", "发改委", "工信部", "证监会", "财政部", "政策", "规划", "监管"),
    "财务数据": ("财报", "年报", "季报", "营收", "净利润", "毛利率", "ROE"),
    "资金行为": ("主力资金", "融资", "北向资金", "成交额", "换手率", "龙虎榜"),
    "财经新闻": ("东方财富", "同花顺", "新浪", "财联社", "证券时报", "新闻"),
}

REPEAT_HINTS = ("转载", "综合", "据悉", "此前", "再度", "继续", "盘中", "市场消息")


def classify_information_source(
    source: str = "",
    title: str = "",
    content: str = "",
    information_type: str | None = None,
    seen_duplicate_keys: set[str] | None = None,
) -> InformationSourceProfile:
    source_only_rank = rank_source(source, "", "")
    rank = source_only_rank if source_only_rank.source_level >= 2 else rank_source(source, title, content)
    text = f"{source} {title} {content}"
    info_type = information_type or infer_information_type(text, rank)
    duplicate_key = make_duplicate_key(text)
    is_repeat = duplicate_key in seen_duplicate_keys if seen_duplicate_keys is not None else has_repeat_hint(text)
    is_original = rank.is_original and not is_repeat

    return InformationSourceProfile(
        source_name=rank.source_name,
        source_level=rank.source_level,
        source_score=rank.source_score,
        is_original=is_original,
        is_repeat=is_repeat,
        information_type=info_type,
        category=rank.category,
        description=rank.description,
        duplicate_key=duplicate_key,
    )


def infer_information_type(text: str, rank: SourceRank | None = None) -> str:
    if rank and rank.source_level >= 5:
        return "公司公告"
    if rank and rank.source_level == 4:
        return "政策信息"
    for info_type, keywords in INFORMATION_TYPE_KEYWORDS.items():
        if any(keyword.lower() in (text or "").lower() for keyword in keywords):
            return info_type
    return "未分类信息"


def has_repeat_hint(text: str) -> bool:
    return any(keyword in (text or "") for keyword in REPEAT_HINTS)


def make_duplicate_key(text: str) -> str:
    normalized = re.sub(r"\s+", "", text or "").lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def can_generate_strong_signal(profile: InformationSourceProfile) -> bool:
    return profile.source_level >= 4 and not profile.is_repeat
