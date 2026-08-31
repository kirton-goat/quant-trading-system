from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRank:
    source_name: str
    source_level: int
    source_score: int
    category: str
    is_original: bool
    description: str


LEVEL_5_KEYWORDS = (
    "上海证券交易所",
    "上交所",
    "深圳证券交易所",
    "深交所",
    "北京证券交易所",
    "北交所",
    "巨潮资讯",
    "公司公告",
    "上市公司公告",
    "监管部门",
    "证监会公告",
)

LEVEL_4_KEYWORDS = (
    "国务院",
    "发改委",
    "国家发展改革委",
    "工信部",
    "工业和信息化部",
    "财政部",
    "央行",
    "中国人民银行",
    "证监会",
    "行业协会",
)

LEVEL_3_KEYWORDS = (
    "财联社",
    "证券时报",
    "上海证券报",
    "中国证券报",
    "第一财经",
    "财新",
    "21世纪经济报道",
)

LEVEL_2_KEYWORDS = (
    "东方财富",
    "同花顺",
    "新浪",
    "新浪财经",
    "腾讯财经",
    "网易财经",
    "界面新闻",
)

LEVEL_1_KEYWORDS = (
    "传闻",
    "网传",
    "自媒体",
    "雪球",
    "股吧",
    "未知",
)


def rank_source(source: str = "", title: str = "", content: str = "") -> SourceRank:
    text = f"{source} {title} {content}"

    if contains_any(text, LEVEL_5_KEYWORDS):
        return SourceRank(
            source_name=source or "权威公告",
            source_level=5,
            source_score=95,
            category="权威公告/监管文件",
            is_original=True,
            description="法律责任明确，可验证性最高",
        )

    if contains_any(text, LEVEL_4_KEYWORDS):
        return SourceRank(
            source_name=source or "政策文件",
            source_level=4,
            source_score=82,
            category="国家/行业政策",
            is_original=True,
            description="政策或行业数据来源，权威性较高",
        )

    if contains_any(text, LEVEL_3_KEYWORDS):
        return SourceRank(
            source_name=source or "财经媒体原创",
            source_level=3,
            source_score=68,
            category="财经媒体原创/调查",
            is_original=True,
            description="主流财经媒体，需继续验证是否为原创",
        )

    if contains_any(text, LEVEL_2_KEYWORDS):
        return SourceRank(
            source_name=source or "财经网站转载",
            source_level=2,
            source_score=45,
            category="财经网站/聚合转载",
            is_original=False,
            description="常见聚合或转载信息，不应直接产生强交易信号",
        )

    if not source or contains_any(text, LEVEL_1_KEYWORDS):
        return SourceRank(
            source_name=source or "未知来源",
            source_level=1,
            source_score=20,
            category="未知/传闻/自媒体",
            is_original=False,
            description="来源不清或传闻性质，默认低权重",
        )

    return SourceRank(
        source_name=source,
        source_level=1,
        source_score=30,
        category="未确认来源",
        is_original=False,
        description="来源未纳入白名单，默认保守处理",
    )


def can_generate_strong_signal(source_rank: SourceRank) -> bool:
    return source_rank.source_level >= 4


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lower = (text or "").lower()
    return any(keyword.lower() in lower for keyword in keywords)
