"""Official announcement and policy source collectors for research records.

The module is intentionally separate from scoring and trading.  It only writes
auditable event records with an original link, publisher, publication time and
fetch metadata.  A source failure or an item without a date is retained in the
sync audit and is never converted into a policy signal.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from announcement_engine import classify_announcement
from event_research_data import EVENT_DATA_DIR, record_research_event
from policy_engine import analyze_policy


REQUEST_TIMEOUT = 20
SYNC_AUDIT_FILE = EVENT_DATA_DIR / "official_source_sync_audit.jsonl"
CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"


@dataclass(frozen=True)
class OfficialSource:
    key: str
    publisher: str
    index_url: str
    event_type: str = "policy"


OFFICIAL_POLICY_SOURCES = (
    OfficialSource("state_council", "国务院/中国政府网", "https://www.gov.cn/zhengce/zhengceku/bmwj/home.htm"),
    OfficialSource("ndrc", "国家发展和改革委员会", "https://www.ndrc.gov.cn/xxgk/zcfb/fzggwl/"),
    OfficialSource("miit", "工业和信息化部", "https://www.miit.gov.cn/zwgk/index.html"),
    OfficialSource("mof", "财政部", "https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/"),
    OfficialSource("csrc", "中国证券监督管理委员会", "https://www.csrc.gov.cn/csrc/c100028/common_list.shtml"),
)


@dataclass(frozen=True)
class OfficialEvent:
    event_type: str
    title: str
    published_at: str
    publisher: str
    source: str
    source_url: str
    content: str = ""
    stock_code: str = ""


def _headers(referer: str = "") -> dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def fetch_cninfo_announcements(days: int = 1, page_size: int = 50, session: requests.Session | None = None) -> list[OfficialEvent]:
    """Fetch recent exchange disclosures through CNInfo's public query endpoint."""
    session = session or requests.Session()
    end = dt.date.today()
    start = end - dt.timedelta(days=max(days, 1))
    payload = {
        "pageNum": 1, "pageSize": max(1, min(page_size, 100)), "column": "szse",
        "tabName": "fulltext", "plate": "", "stock": "", "searchkey": "", "secid": "",
        "category": "", "trade": "", "seDate": f"{start:%Y-%m-%d}~{end:%Y-%m-%d}",
        "sortName": "", "sortType": "", "isHLtitle": "true",
    }
    response = session.post(CNINFO_QUERY_URL, data=payload, headers=_headers("https://www.cninfo.com.cn/"), timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    items = response.json().get("announcements", [])
    events: list[OfficialEvent] = []
    for item in items:
        title = _clean(item.get("announcementTitle", ""))
        adjunct = str(item.get("adjunctUrl", "")).strip()
        timestamp = item.get("announcementTime")
        published = _timestamp_to_iso(timestamp)
        if not title or not adjunct or not published:
            continue
        code = re.sub(r"\D", "", str(item.get("secCode", ""))).zfill(6)[-6:]
        url = urljoin("https://static.cninfo.com.cn/", adjunct.lstrip("/"))
        company = _clean(item.get("secName", ""))
        events.append(OfficialEvent(
            event_type=f"公司公告/{classify_announcement(title)}", title=title, published_at=published,
            publisher="巨潮资讯/交易所法定信息披露", source="巨潮资讯（交易所公告）", source_url=url,
            content=f"公司：{company}；股票代码：{code}" if company or code else "", stock_code=code,
        ))
    return events


def fetch_official_policy_source(source: OfficialSource, limit: int = 40, session: requests.Session | None = None) -> list[OfficialEvent]:
    """Parse dated items from a source's official policy listing page.

    The parsers intentionally accept only links that carry a recognizable
    publication date in their list block. This sacrifices coverage rather than
    inventing an event time.
    """
    session = session or requests.Session()
    response = session.get(source.index_url, headers=_headers(source.index_url), timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    # These government sites sometimes advertise ISO-8859-1 despite serving
    # UTF-8. Prefer the document's UTF-8 bytes so Chinese titles are not
    # silently persisted as mojibake.
    response.encoding = "utf-8"
    rows = _extract_dated_links(response.text, source.index_url, limit)
    events: list[OfficialEvent] = []
    for title, published_at, url in rows:
        policy = analyze_policy(title, title, source=source.publisher)
        content = f"政策类型：{policy.policy_type}；影响行业：{','.join(policy.affected_industries)}；影响周期：{policy.duration}；利好等级：{policy.positive_level}"
        events.append(OfficialEvent(
            event_type=f"政策信息/{policy.policy_type}", title=title, published_at=published_at,
            publisher=source.publisher, source=source.publisher, source_url=url, content=content,
        ))
    return events


def sync_official_sources(days: int = 1, policy_limit: int = 40) -> dict[str, Any]:
    EVENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "sources": [],
        "saved": 0, "duplicates": 0, "failures": 0,
    }
    jobs: list[tuple[str, str, Any]] = [("cninfo", "announcement", lambda: fetch_cninfo_announcements(days=days))]
    jobs.extend((source.key, "policy", lambda source=source: fetch_official_policy_source(source, policy_limit)) for source in OFFICIAL_POLICY_SOURCES)
    for key, kind, fetcher in jobs:
        try:
            events = fetcher()
            saved = 0
            duplicates = 0
            for event in events:
                recorded = record_research_event(
                    event.event_type, event.published_at, event.stock_code, 50.0,
                    event.source, event.title, event.content, publisher=event.publisher,
                    source_url=event.source_url, source_kind=f"official_{kind}", is_official=True,
                )
                saved += int(recorded)
                duplicates += int(not recorded)
            audit["saved"] += saved
            audit["duplicates"] += duplicates
            audit["sources"].append({"source": key, "status": "ok", "fetched": len(events), "saved": saved, "duplicates": duplicates})
        except Exception as error:
            audit["failures"] += 1
            audit["sources"].append({"source": key, "status": "failed", "error": str(error)})
    audit["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with SYNC_AUDIT_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    return audit


def _extract_dated_links(html: str, base_url: str, limit: int) -> list[tuple[str, str, str]]:
    # Government CMS list pages vary by agency. Work per list-like block so a
    # date adjacent to a title remains tied to that link rather than to page chrome.
    blocks = re.findall(r"<(?:li|tr|div)[^>]*>(.*?)</(?:li|tr|div)>", html, flags=re.IGNORECASE | re.DOTALL)
    matches: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for block in blocks:
        date = _extract_date(_strip_tags(block))
        link = re.search(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", block, flags=re.IGNORECASE | re.DOTALL)
        if not date or not link:
            continue
        href = link.group(1).strip()
        title = _clean(_strip_tags(link.group(2)))
        url = urljoin(base_url, href)
        if not title or not _is_http_url(url) or url in seen:
            continue
        seen.add(url)
        matches.append((title, date, url))
        if len(matches) >= limit:
            break
    if matches:
        return matches

    # NDRC policy lists place an auxiliary nested <div> before the date. The
    # generic block expression then closes too early; bind each anchor to the
    # next date only within the same list item instead of using page-level text.
    list_items = re.findall(r"<li[^>]*>(.*?)</li>", html, flags=re.IGNORECASE | re.DOTALL)
    for item in list_items:
        date = _extract_date(_strip_tags(item))
        link = re.search(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", item, flags=re.IGNORECASE | re.DOTALL)
        if not date or not link:
            continue
        href = link.group(1).strip()
        title = _clean(_strip_tags(link.group(2)))
        url = urljoin(base_url, href)
        if not title or not _is_http_url(url) or url in seen:
            continue
        seen.add(url)
        matches.append((title, date, url))
        if len(matches) >= limit:
            break
    return matches


def _timestamp_to_iso(value: Any) -> str:
    try:
        return dt.datetime.fromtimestamp(float(value) / 1000, tz=dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return ""


def _extract_date(value: str) -> str:
    match = re.search(r"(20\d{2})[./年-](\d{1,2})[./月-](\d{1,2})", value)
    if not match:
        return ""
    return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", unescape(value or ""))


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _is_http_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync official announcements and policy sources into research-only event storage.")
    parser.add_argument("--days", type=int, default=1, help="CNInfo announcement lookback window")
    parser.add_argument("--policy-limit", type=int, default=40, help="Maximum records parsed from each policy source")
    args = parser.parse_args()
    print(json.dumps(sync_official_sources(days=args.days, policy_limit=args.policy_limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
