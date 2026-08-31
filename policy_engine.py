from __future__ import annotations

import datetime as dt
import contextlib
import io
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import akshare as ak

from source_rank import SourceRank, rank_source


@dataclass
class PolicyItem:
    title: str
    content: str
    published_at: str
    source: str
    policy_type: str
    affected_industries: list[str] = field(default_factory=list)
    affected_stocks: list[str] = field(default_factory=list)
    duration: str = "未知"
    positive_level: int = 0
    source_rank: SourceRank | None = None

    def to_news_like(self) -> dict[str, Any]:
        rank = self.source_rank or rank_source(self.source, self.title, self.content)
        return {
            "title": self.title,
            "content": self.content,
            "published_at": self.published_at,
            "source": self.source,
            "source_level": rank.source_level,
            "source_score": rank.source_score,
            "policy_type": self.policy_type,
            "affected_industries": self.affected_industries,
            "affected_stocks": self.affected_stocks,
            "duration": self.duration,
            "positive_level": self.positive_level,
        }


POLICY_SOURCES = ("国务院", "发改委", "工信部", "财政部", "央行", "证监会", "行业协会")

INDUSTRY_KEYWORDS = {
    "人工智能": ("人工智能", "AI", "大模型", "算力", "数据要素"),
    "新能源汽车": ("新能源汽车", "智能网联", "充电桩", "动力电池", "汽车"),
    "半导体": ("半导体", "芯片", "集成电路", "国产替代"),
    "医药": ("医药", "创新药", "医疗", "医保", "器械"),
    "房地产": ("房地产", "住房", "城中村", "保障房"),
    "军工": ("军工", "国防", "航空航天"),
    "能源": ("能源", "电力", "光伏", "风电", "储能", "煤炭"),
    "消费": ("消费", "零售", "家电", "食品饮料"),
}

POSITIVE_KEYWORDS = ("支持", "鼓励", "加快", "提升", "补贴", "减税", "建设", "扩大", "推动")
NEGATIVE_KEYWORDS = ("限制", "禁止", "整治", "处罚", "压降", "风险", "整改", "监管趋严")
LONG_TERM_KEYWORDS = ("规划", "纲要", "五年", "长期", "到2030", "到2027")
SHORT_TERM_KEYWORDS = ("即日起", "近期", "临时", "阶段性", "本月", "下月")


def analyze_policy(title: str, content: str, source: str = "政策文件") -> PolicyItem:
    text = f"{title} {content}"
    rank = rank_source(source, title, content)
    industries = detect_industries(text)
    policy_type = classify_policy(text)
    duration = estimate_duration(text)
    positive_level = estimate_positive_level(text)
    return PolicyItem(
        title=title,
        content=content,
        published_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source=source,
        policy_type=policy_type,
        affected_industries=industries,
        duration=duration,
        positive_level=positive_level,
        source_rank=rank,
    )


def fetch_policy_news(timeout: int = 20) -> list[PolicyItem]:
    """Fetch policy-like public news from available akshare news feeds.

    This is a research input layer. It does not execute trades and should be
    treated as a fallback until direct ministry/official-site connectors are
    added.
    """
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_fetch_policy_news_worker, args=(queue,))
    process.daemon = True
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        print(f"政策新闻接口超过 {timeout} 秒未响应，已跳过")
        return []

    if queue.empty():
        return []
    status, payload = queue.get()
    if status == "error":
        print(f"政策新闻接口失败: {payload}")
        return []
    return payload


def _fetch_policy_news_worker(queue: mp.Queue) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = ak.stock_info_global_em()
        items: list[PolicyItem] = []
        for _, row in df.head(100).iterrows():
            title = first_existing(row, ("标题",))
            content = first_existing(row, ("摘要", "内容")) or title
            published_at = first_existing(row, ("发布时间", "时间"))
            text = f"{title} {content}"
            if not any(keyword in text for keyword in POLICY_SOURCES):
                continue
            item = analyze_policy(title, content, source="东方财富政策新闻")
            item.published_at = published_at or item.published_at
            items.append(item)
        queue.put(("ok", items))
    except Exception as exc:
        queue.put(("error", str(exc)))


def classify_policy(text: str) -> str:
    if "监管" in text or "处罚" in text or "整治" in text or "规范" in text:
        return "监管变化"
    if "补贴" in text or "贴息" in text or "减税" in text or "税收优惠" in text:
        return "补贴政策"
    if "规划" in text or "纲要" in text or "方案" in text or "行动计划" in text:
        return "产业规划"
    if "支持" in text or "鼓励" in text or "推动" in text or "加快" in text:
        return "行业支持政策"
    if "数据" in text or "统计" in text or "协会" in text:
        return "行业数据"
    return "政策信息"


def detect_industries(text: str) -> list[str]:
    industries: list[str] = []
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            industries.append(industry)
    return industries


def estimate_duration(text: str) -> str:
    if any(keyword in text for keyword in LONG_TERM_KEYWORDS):
        return "中长期"
    if any(keyword in text for keyword in SHORT_TERM_KEYWORDS):
        return "短期"
    return "中期"


def estimate_positive_level(text: str) -> int:
    score = 0
    score += sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text)
    score -= sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text)
    if score >= 3:
        return 5
    if score >= 1:
        return 4
    if score == 0:
        return 3
    if score == -1:
        return 2
    return 1


def first_existing(row: Any, names: tuple[str, ...]) -> str:
    for name in names:
        if name in row.index and row[name] is not None:
            value = str(row[name]).strip()
            if value and value.lower() != "nan":
                return value
    return ""
