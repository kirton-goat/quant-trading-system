from __future__ import annotations

import datetime as dt
import contextlib
import io
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import akshare as ak

from source_rank import SourceRank, rank_source


@dataclass
class AnnouncementItem:
    company_name: str
    stock_code: str
    published_at: str
    title: str
    content: str
    announcement_type: str
    source: str = "巨潮资讯/交易所公告"
    source_rank: SourceRank | None = None

    def to_news_like(self) -> dict[str, Any]:
        rank = self.source_rank or rank_source(self.source, self.title, self.content)
        return {
            "title": self.title,
            "content": self.content or self.title,
            "published_at": self.published_at,
            "source": self.source,
            "source_level": rank.source_level,
            "source_score": rank.source_score,
            "announcement_type": self.announcement_type,
            "stock_code": self.stock_code,
            "company_name": self.company_name,
        }


ANNOUNCEMENT_CATEGORIES = {
    "重大合同": ("重大合同", "合同", "中标", "订单", "采购", "销售协议"),
    "业绩预告": ("业绩预告", "预增", "预减", "扭亏", "业绩快报"),
    "财务变化": ("净利润", "营收", "毛利率", "资产减值", "财务报告", "年报", "季报"),
    "并购重组": ("并购", "重组", "收购", "资产置换", "重大资产"),
    "股权变化": ("股权", "股份", "减持", "增持", "质押", "解禁", "持股"),
    "回购": ("回购", "股份回购", "注销股份"),
    "风险提示": ("风险提示", "退市", "处罚", "诉讼", "仲裁", "违规", "亏损"),
    "澄清公告": ("澄清", "说明公告", "异动公告", "媒体报道"),
    "经营变化": ("经营", "产能", "量产", "投产", "停产", "涨价", "交付"),
}


def classify_announcement(title: str, content: str = "") -> str:
    text = f"{title} {content}"
    for category, keywords in ANNOUNCEMENT_CATEGORIES.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "其他公告"


def fetch_company_announcements(
    stock_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    keyword: str = "",
    timeout: int = 20,
) -> list[AnnouncementItem]:
    start_date = start_date or (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y%m%d")
    end_date = end_date or dt.datetime.now().strftime("%Y%m%d")
    return _call_with_timeout(stock_code, start_date, end_date, keyword, timeout)


def _call_with_timeout(
    stock_code: str,
    start_date: str,
    end_date: str,
    keyword: str,
    timeout: int,
) -> list[AnnouncementItem]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_fetch_company_announcements_worker,
        args=(queue, stock_code, start_date, end_date, keyword),
    )
    process.daemon = True
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        print(f"公告接口超过 {timeout} 秒未响应，已跳过")
        return []

    if queue.empty():
        return []
    status, payload = queue.get()
    if status == "error":
        print(f"公告接口失败: {payload}")
        return []
    return payload


def _fetch_company_announcements_worker(
    queue: mp.Queue,
    stock_code: str,
    start_date: str,
    end_date: str,
    keyword: str,
) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = ak.stock_zh_a_disclosure_report_cninfo(
                symbol=stock_code,
                market="沪深京",
                keyword=keyword,
                start_date=start_date,
                end_date=end_date,
            )
        items: list[AnnouncementItem] = []
        for _, row in df.head(50).iterrows():
            title = first_existing(row, ("公告标题", "标题", "announcementTitle"))
            published_at = first_existing(row, ("公告时间", "公告日期", "发布日期", "time"))
            company_name = first_existing(row, ("证券简称", "公司简称", "简称", "companyName"))
            code = first_existing(row, ("证券代码", "代码", "stockCode")) or stock_code
            content = first_existing(row, ("公告内容", "内容", "摘要")) or title
            category = classify_announcement(title, content)
            source = "上市公司公告/巨潮资讯"
            items.append(
                AnnouncementItem(
                    company_name=company_name,
                    stock_code=code,
                    published_at=published_at,
                    title=title,
                    content=content,
                    announcement_type=category,
                    source=source,
                    source_rank=rank_source(source, title, content),
                )
            )
        queue.put(("ok", items))
    except Exception as exc:
        queue.put(("error", str(exc)))


def fetch_market_announcements(date: str | None = None, timeout: int = 20) -> list[AnnouncementItem]:
    date = date or dt.datetime.now().strftime("%Y%m%d")
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_fetch_market_announcements_worker, args=(queue, date))
    process.daemon = True
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        print(f"公告大全接口超过 {timeout} 秒未响应，已跳过")
        return []
    if queue.empty():
        return []
    status, payload = queue.get()
    if status == "error":
        print(f"公告大全接口失败: {payload}")
        return []
    return payload


def _fetch_market_announcements_worker(queue: mp.Queue, date: str) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = ak.stock_notice_report(symbol="全部", date=date)
        items: list[AnnouncementItem] = []
        for _, row in df.head(100).iterrows():
            title = first_existing(row, ("公告标题", "标题"))
            company_name = first_existing(row, ("名称", "证券简称", "公司简称"))
            code = first_existing(row, ("代码", "证券代码"))
            published_at = first_existing(row, ("公告时间", "公告日期", "日期")) or date
            content = first_existing(row, ("内容", "摘要")) or title
            source = "上市公司公告/东方财富公告大全"
            items.append(
                AnnouncementItem(
                    company_name=company_name,
                    stock_code=code,
                    published_at=published_at,
                    title=title,
                    content=content,
                    announcement_type=classify_announcement(title, content),
                    source=source,
                    source_rank=rank_source("公司公告", title, content),
                )
            )
        queue.put(("ok", items))
    except Exception as exc:
        queue.put(("error", str(exc)))


def first_existing(row: Any, names: tuple[str, ...]) -> str:
    for name in names:
        if name in row.index and row[name] is not None:
            value = str(row[name]).strip()
            if value and value.lower() != "nan":
                return value
    return ""
