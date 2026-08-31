from __future__ import annotations

from dataclasses import dataclass

from announcement_engine import fetch_market_announcements
from market import NewsItem, fetch_latest_news
from policy_engine import fetch_policy_news


@dataclass
class EventSourceItem:
    news: NewsItem
    forced_code: str | None = None
    event_type: str = "财经新闻"


def fetch_latest_event(timeout: int = 20, exclude_titles: set[str] | None = None) -> EventSourceItem | None:
    exclude_titles = exclude_titles or set()
    per_source_timeout = max(5, min(timeout, 10))

    for item in fetch_announcement_events(per_source_timeout):
        if item.news.title not in exclude_titles:
            return item

    for item in fetch_policy_events(per_source_timeout):
        if item.news.title not in exclude_titles:
            return item

    news = fetch_latest_news(timeout=timeout)
    if news is None or news.title in exclude_titles:
        return None
    return EventSourceItem(news=news, event_type="财经新闻")


def fetch_announcement_events(timeout: int = 8) -> list[EventSourceItem]:
    items: list[EventSourceItem] = []
    for announcement in fetch_market_announcements(timeout=timeout):
        title = announcement.title or ""
        if not title:
            continue
        content = build_announcement_content(announcement)
        news = NewsItem(
            title=title,
            content=content,
            published_at=announcement.published_at,
            source=announcement.source,
        )
        items.append(
            EventSourceItem(
                news=news,
                forced_code=normalize_stock_code(announcement.stock_code),
                event_type=f"公司公告/{announcement.announcement_type}",
            )
        )
    return items


def fetch_policy_events(timeout: int = 8) -> list[EventSourceItem]:
    items: list[EventSourceItem] = []
    for policy in fetch_policy_news(timeout=timeout):
        title = policy.title or ""
        if not title:
            continue
        content = build_policy_content(policy)
        news = NewsItem(
            title=title,
            content=content,
            published_at=policy.published_at,
            source=policy.source,
        )
        items.append(EventSourceItem(news=news, event_type=f"政策信息/{policy.policy_type}"))
    return items


def build_announcement_content(announcement) -> str:
    parts = [
        announcement.content or announcement.title,
        f"公司：{announcement.company_name}" if announcement.company_name else "",
        f"股票代码：{announcement.stock_code}" if announcement.stock_code else "",
        f"公告类型：{announcement.announcement_type}" if announcement.announcement_type else "",
    ]
    return "；".join(part for part in parts if part)


def build_policy_content(policy) -> str:
    parts = [
        policy.content or policy.title,
        f"政策类型：{policy.policy_type}" if policy.policy_type else "",
        f"影响行业：{','.join(policy.affected_industries)}" if policy.affected_industries else "",
        f"影响周期：{policy.duration}" if policy.duration else "",
        f"利好等级：{policy.positive_level}" if policy.positive_level else "",
    ]
    return "；".join(part for part in parts if part)


def normalize_stock_code(value: str | None) -> str | None:
    if not value:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    return None
