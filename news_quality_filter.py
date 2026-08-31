from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from market import MarketSnapshot, NewsItem
from source_manager import classify_information_source


HIGH_TRUST_SOURCES = ("交易所", "上交所", "深交所", "北交所", "公告", "公司公告")
POLICY_SOURCES = ("政府", "国务院", "发改委", "工信部", "财政部", "央行", "证监会")
ORIGINAL_MEDIA = ("财联社", "证券时报", "上海证券报", "中国证券报", "第一财经")
GENERAL_MEDIA = ("东方财富", "同花顺", "新浪", "腾讯", "网易", "界面")

FUNDAMENTAL_STRONG = (
    "订单",
    "中标",
    "合同",
    "收入",
    "营收",
    "利润",
    "净利润",
    "业绩",
    "预增",
    "量产",
    "交付",
    "涨价",
    "产能",
)
FUNDAMENTAL_WEAK = (
    "战略合作",
    "框架协议",
    "意向",
    "规划",
    "概念",
    "生态",
    "探索",
    "布局",
    "发布会",
    "传闻",
)
HYPE_KEYWORDS = (
    "AI",
    "大模型",
    "算力",
    "机器人",
    "低空经济",
    "DeepSeek",
    "华为",
    "小米汽车",
    "问界",
)
NEGATIVE_FUNDAMENTAL_PHRASES = (
    "未涉及订单",
    "未涉及收入",
    "未涉及利润",
    "未产生收入",
    "未产生利润",
    "尚未产生收入",
    "尚未产生利润",
    "暂无订单",
    "暂无收入",
    "暂无利润",
    "未形成订单",
    "未形成收入",
)


@dataclass
class NewsQualityReport:
    source_level: int
    source_score: float
    novelty_score: float
    fundamental_score: float
    market_reaction_score: float
    trade_value_score: float
    quality_score: float
    technical_score: float
    information_type: str = "未分类信息"
    is_original: bool = False
    is_repeat: bool = False
    has_financial_impact: bool = False
    market_already_priced: bool = False
    trade_value: str = "medium"
    risk: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    duplicate_key: str = ""
    is_duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_level": self.source_level,
            "source_score": round(self.source_score, 2),
            "novelty_score": round(self.novelty_score, 2),
            "fundamental_score": round(self.fundamental_score, 2),
            "market_reaction_score": round(self.market_reaction_score, 2),
            "trade_value_score": round(self.trade_value_score, 2),
            "quality_score": round(self.quality_score, 2),
            "technical_score": round(self.technical_score, 2),
            "information_type": self.information_type,
            "is_original": self.is_original,
            "is_repeat": self.is_repeat,
            "has_financial_impact": self.has_financial_impact,
            "market_already_priced": self.market_already_priced,
            "trade_value": self.trade_value,
            "risk": self.risk,
            "reasons": self.reasons,
            "duplicate_key": self.duplicate_key,
            "is_duplicate": self.is_duplicate,
        }


class NewsQualityFilter:
    def __init__(self, log_file: Path | None = None) -> None:
        self.log_file = log_file
        self.seen_keys: set[str] = set()
        self.seen_titles: list[str] = []
        if log_file:
            self._load_history(log_file)

    def evaluate(
        self,
        news: NewsItem,
        mapped_target: str | None = None,
        snapshot: MarketSnapshot | None = None,
    ) -> NewsQualityReport:
        text = f"{news.title} {news.content}"
        duplicate_key = make_duplicate_key(text)
        is_duplicate = duplicate_key in self.seen_keys or self._is_similar_to_history(news.title)

        source_profile = classify_information_source(news.source, news.title, news.content, seen_duplicate_keys=self.seen_keys)
        source_score = source_profile.source_score / 100
        source_reason = f"{source_profile.category}: {source_profile.description}"
        novelty_score, novelty_reason = score_novelty(is_duplicate, news.source, news.title)
        fundamental_score, fundamental_reason, fundamental_risks = score_fundamental(text)
        market_score, market_reason, market_risks = score_market_reaction(snapshot)
        technical_score = score_technical(snapshot)
        market_already_priced = market_score < 0.45
        has_financial_impact = fundamental_score >= 0.65

        risk: list[str] = []
        reasons = [source_reason, novelty_reason, fundamental_reason, market_reason]
        risk.extend(fundamental_risks)
        risk.extend(market_risks)

        if is_duplicate:
            risk.append("消息重复风险")
        if contains_any(text, HYPE_KEYWORDS) and fundamental_score < 0.55:
            risk.append("概念炒作风险")
        if market_score < 0.45 and fundamental_score < 0.65:
            risk.append("热点追涨风险")

        trade_value_score = weighted_average(
            [
                (source_score, 0.25),
                (novelty_score, 0.2),
                (fundamental_score, 0.2),
                (market_score, 0.2),
                (technical_score, 0.15),
            ]
        )
        quality_score = trade_value_score
        trade_value = classify_trade_value(trade_value_score)

        if source_profile.source_level <= 2 and trade_value_score >= 0.7:
            risk.append("低等级来源强信号抑制")
            trade_value_score = min(trade_value_score, 0.62)
            quality_score = min(quality_score, 0.62)
            trade_value = classify_trade_value(trade_value_score)

        report = NewsQualityReport(
            source_level=source_profile.source_level,
            source_score=source_score,
            novelty_score=novelty_score,
            fundamental_score=fundamental_score,
            market_reaction_score=market_score,
            trade_value_score=trade_value_score,
            quality_score=quality_score,
            technical_score=technical_score,
            information_type=source_profile.information_type,
            is_original=source_profile.is_original and not is_duplicate,
            is_repeat=is_duplicate,
            has_financial_impact=has_financial_impact,
            market_already_priced=market_already_priced,
            trade_value=trade_value,
            risk=dedupe_preserve_order(risk),
            reasons=[x for x in reasons if x],
            duplicate_key=duplicate_key,
            is_duplicate=is_duplicate,
        )

        self.seen_keys.add(duplicate_key)
        if news.title:
            self.seen_titles.append(normalize_text(news.title))
        return report

    def _load_history(self, log_file: Path) -> None:
        if not log_file.exists():
            return
        try:
            with log_file.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    title = row.get("新闻标题") or row.get("鏂伴椈鏍囬") or ""
                    if title:
                        normalized = normalize_text(title)
                        self.seen_titles.append(normalized)
                        self.seen_keys.add(make_duplicate_key(title))
        except Exception:
            return

    def _is_similar_to_history(self, title: str) -> bool:
        current = normalize_text(title)
        if not current:
            return False
        for old in self.seen_titles[-500:]:
            if not old:
                continue
            if current in old or old in current:
                return True
            if jaccard_similarity(current, old) >= 0.72:
                return True
        return False


def make_duplicate_key(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def score_source(source: str, title: str, content: str) -> tuple[float, str]:
    text = f"{source} {title} {content}"
    if contains_any(text, HIGH_TRUST_SOURCES):
        return 0.95, "来源接近交易所/公司公告，可信度高"
    if contains_any(text, POLICY_SOURCES):
        return 0.85, "来源包含政府/监管政策信息，可信度较高"
    if contains_any(source, ORIGINAL_MEDIA):
        return 0.72, "来源为财经媒体，可信度中高"
    if contains_any(source, GENERAL_MEDIA):
        return 0.58, "来源为财经网站或聚合平台，需防转载"
    if source and source != "未知":
        return 0.48, "来源可识别但权威性一般"
    return 0.28, "未知来源，可信度低"


def score_novelty(is_duplicate: bool, source: str, title: str) -> tuple[float, str]:
    if is_duplicate:
        return 0.25, "历史日志中存在相似标题，可能为重复报道"
    if contains_any(title, ("再度", "继续", "转载", "据悉", "此前", "盘中")):
        return 0.48, "标题暗示可能是延续性或转载信息"
    if source in {"东方财富", "同花顺", "新浪"}:
        return 0.62, "来自聚合平台，初步视为中等新颖性"
    return 0.72, "未发现明显重复，具备一定新颖性"


def score_fundamental(text: str) -> tuple[float, str, list[str]]:
    strong_hits = keyword_hits(text, FUNDAMENTAL_STRONG)
    weak_hits = keyword_hits(text, FUNDAMENTAL_WEAK)
    risks: list[str] = []

    if contains_any(text, NEGATIVE_FUNDAMENTAL_PHRASES) or has_negative_fundamental_context(text):
        risks.append("概念炒作风险")
        return 0.32, "文本明确缺少订单/收入/利润等财务兑现证据", risks
    if strong_hits >= 2:
        return 0.82, "新闻涉及订单/业绩/收入等较强基本面要素", risks
    if strong_hits == 1 and weak_hits == 0:
        return 0.68, "新闻包含一个基本面要素，但仍需验证持续性", risks
    if weak_hits > 0 and strong_hits == 0:
        risks.append("概念炒作风险")
        return 0.38, "新闻偏战略规划/概念合作，缺少业绩兑现证据", risks
    if contains_any(text, HYPE_KEYWORDS):
        risks.append("概念炒作风险")
        return 0.42, "新闻与热点概念相关，但缺少直接基本面支撑", risks
    return 0.5, "基本面关联不明确", risks


def score_market_reaction(snapshot: MarketSnapshot | None) -> tuple[float, str, list[str]]:
    if snapshot is None:
        return 0.5, "无具体股票行情，无法判断市场是否提前反应", []

    risks: list[str] = []
    change = safe_float(snapshot.change_pct)
    pct_change_20d = safe_float(snapshot.pct_change_20d)
    pct_change_60d = safe_float(getattr(snapshot, "pct_change_60d", None))
    volume_ratio_5_20 = safe_float(snapshot.volume_ratio_5_20)
    gap_pct = safe_float(snapshot.gap_pct)
    turnover_rate = safe_float(getattr(snapshot, "turnover_rate", None))
    trend = snapshot.trend or ""
    summary = snapshot.summary or ""

    if pct_change_20d is not None and pct_change_20d >= 30:
        risks.append("高位利好兑现风险")
        return 0.22, f"近20日涨幅{pct_change_20d}%，消息可能已被提前交易", risks
    if pct_change_20d is not None and pct_change_20d >= 18:
        risks.append("热点追涨风险")
        return 0.34, f"近20日涨幅{pct_change_20d}%，追涨风险较高", risks
    if pct_change_60d is not None and pct_change_60d >= 50:
        risks.append("中期高位风险")
        return 0.32, f"近60日涨幅{pct_change_60d}%，中期高位兑现风险较高", risks
    if gap_pct is not None and gap_pct >= 5:
        risks.append("高开兑现风险")
        return 0.36, f"跳空幅度{gap_pct}%，存在高开兑现风险", risks
    if turnover_rate is not None and turnover_rate >= 18:
        risks.append("高换手博弈风险")
        return 0.42, f"换手率{turnover_rate}%，短线博弈较重", risks
    if change is not None and change >= 7:
        risks.append("高位利好兑现风险")
        return 0.25, "当日或最近数据涨幅较大，可能已提前反应", risks
    if change is not None and change >= 4:
        risks.append("热点追涨风险")
        return 0.38, "涨幅偏高，存在追涨风险", risks
    if "空头" in trend:
        return 0.45, "技术趋势偏弱，新闻驱动需要更高确认度", risks
    if volume_ratio_5_20 is not None and volume_ratio_5_20 >= 2.5 and fundamental_score_hint(snapshot) < 0.65:
        risks.append("异常放量追涨风险")
        return 0.4, f"5/20日量能比{volume_ratio_5_20}，需防短线资金提前交易", risks
    if "多头" in trend and "放量" in summary:
        return 0.72, "趋势与量能配合较好，市场反应相对健康", risks
    if "多头" in trend:
        return 0.64, "趋势偏强，但仍需确认是否已被提前交易", risks
    return 0.55, "市场反应中性", risks


def fundamental_score_hint(snapshot: MarketSnapshot | None) -> float:
    return 0.5


def score_technical(snapshot: MarketSnapshot | None) -> float:
    if snapshot is None:
        return 0.5
    trend = snapshot.trend or ""
    summary = snapshot.summary or ""
    score = 0.5
    if "多头" in trend:
        score += 0.18
    if "空头" in trend:
        score -= 0.18
    if "放量" in summary:
        score += 0.08
    if "超买" in summary:
        score -= 0.12
    if "超卖" in summary:
        score += 0.05
    return clamp(score)


def combine_scores(quality_score: float, ai_score: Any, technical_score: float) -> float:
    ai_normalized = clamp(safe_float(ai_score, default=0) / 10)
    combined = quality_score * ai_normalized * technical_score
    return round(combined * 10, 2)


def classify_trade_value(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.48:
        return "medium"
    return "low"


def weighted_average(items: list[tuple[float, float]]) -> float:
    weight_sum = sum(weight for _, weight in items)
    if weight_sum == 0:
        return 0
    return clamp(sum(score * weight for score, weight in items) / weight_sum)


def keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword.lower() in text.lower())


def has_negative_fundamental_context(text: str) -> bool:
    return bool(re.search(r"(未涉及|未形成|未产生|尚未|暂无).{0,12}(订单|收入|营收|利润|净利润)", text or ""))


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def jaccard_similarity(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def tokenize(text: str) -> list[str]:
    if re.search(r"[a-zA-Z0-9]", text):
        return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}", text.lower())
    return [text[i : i + 2] for i in range(max(len(text) - 1, 0))]


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result
