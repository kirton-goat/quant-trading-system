from __future__ import annotations

import contextlib
import csv
import datetime as dt
import io
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT = BASE_DIR / "sector_rank.md"
DEFAULT_LOG = BASE_DIR / "logs" / "trading_log.csv"


@dataclass
class SectorRank:
    rank: int
    sector_name: str
    total_score: float
    change_pct: float | None = None
    volume_change: float | None = None
    money_strength: float | None = None
    news_heat: int = 0
    reason: str = ""
    risk_tags: list[str] = field(default_factory=list)

    def to_row(self) -> list[Any]:
        return [
            self.rank,
            self.sector_name,
            self.total_score,
            self.change_pct,
            self.volume_change,
            self.money_strength,
            self.news_heat,
            self.reason,
            " | ".join(self.risk_tags),
        ]


INDUSTRY_NEWS_KEYWORDS = {
    "半导体": ("半导体", "芯片", "集成电路", "光刻机"),
    "人工智能": ("人工智能", "AI", "大模型", "算力", "数据要素", "DeepSeek"),
    "机器人": ("机器人", "人形机器人", "减速器", "伺服"),
    "汽车整车": ("汽车", "整车", "问界", "小米汽车", "新能源汽车"),
    "汽车零部件": ("零部件", "汽配", "智能驾驶", "激光雷达"),
    "光伏设备": ("光伏", "硅片", "组件", "逆变器"),
    "电池": ("电池", "锂电", "固态电池", "储能"),
    "医药商业": ("医药", "创新药", "医疗", "医保"),
    "石油行业": ("石油", "油价", "原油", "天然气"),
    "有色金属": ("有色", "铜", "铝", "锂", "黄金"),
    "证券": ("券商", "证券", "资本市场"),
    "银行": ("银行", "息差", "信贷"),
    "房地产开发": ("房地产", "地产", "住房", "城中村"),
    "消费电子": ("消费电子", "手机", "折叠屏", "苹果"),
}


def build_sector_rotation_report(
    output_file: Path = DEFAULT_REPORT,
    log_file: Path = DEFAULT_LOG,
    timeout: int = 30,
    top_n: int = 30,
) -> list[SectorRank]:
    rankings = rank_sectors(timeout=timeout, top_n=top_n, log_file=log_file)
    write_sector_rank_report(rankings, output_file)
    return rankings


def rank_sectors(timeout: int = 30, top_n: int = 30, log_file: Path = DEFAULT_LOG) -> list[SectorRank]:
    payload = _call_with_timeout(_fetch_sector_payload, timeout=timeout)
    if not isinstance(payload, dict):
        rankings = [
            SectorRank(
                rank=1,
                sector_name="数据不足",
                total_score=50,
                reason=f"行业接口不可用：{payload}",
                risk_tags=["行业数据不足"],
            )
        ]
        return rankings

    spot_df = payload.get("spot")
    flow_df = payload.get("flow")
    heat = calculate_news_heat(log_file)
    merged = merge_sector_data(spot_df, flow_df, heat)
    rankings = score_sectors(merged)
    return rankings[:top_n]


def _fetch_sector_payload() -> dict[str, pd.DataFrame]:
    spot_df = pd.DataFrame()
    flow_df = pd.DataFrame()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spot_df = ak.stock_board_industry_name_em()
    except Exception:
        spot_df = pd.DataFrame()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            flow_df = ak.stock_fund_flow_industry(symbol="即时")
    except Exception:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                flow_df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        except Exception:
            flow_df = pd.DataFrame()
    return {"spot": spot_df, "flow": flow_df}


def merge_sector_data(spot_df: pd.DataFrame | None, flow_df: pd.DataFrame | None, heat: dict[str, int]) -> list[dict[str, Any]]:
    sectors: dict[str, dict[str, Any]] = {}

    if spot_df is not None and not spot_df.empty:
        for _, row in spot_df.iterrows():
            name = first_existing(row, ("板块名称", "名称", "行业名称"))
            if not name:
                continue
            sectors[name] = {
                "sector_name": name,
                "change_pct": to_float(first_existing(row, ("涨跌幅", "涨幅", "今日涨跌幅", "行业-涨跌幅"))),
                "volume_change": to_float(first_existing(row, ("换手率", "成交量", "成交额", "流入资金", "流出资金"))),
                "money_strength": None,
                "news_heat": sector_heat(name, heat),
            }

    if flow_df is not None and not flow_df.empty:
        for _, row in flow_df.iterrows():
            name = first_existing(row, ("行业", "名称", "板块名称"))
            if not name:
                continue
            item = sectors.setdefault(
                name,
                {
                    "sector_name": name,
                    "change_pct": None,
                    "volume_change": None,
                    "money_strength": None,
                    "news_heat": sector_heat(name, heat),
                },
            )
            item["money_strength"] = to_float(
                first_existing(
                    row,
                    (
                        "主力净流入-净额",
                        "今日主力净流入-净额",
                        "净流入",
                        "主力净流入",
                        "今日主力净流入",
                        "净额",
                    ),
                )
            )
            flow_in = to_float(first_existing(row, ("流入资金", "今日流入资金")))
            flow_out = to_float(first_existing(row, ("流出资金", "今日流出资金")))
            if item.get("volume_change") is None and (flow_in is not None or flow_out is not None):
                item["volume_change"] = round((flow_in or 0) + (flow_out or 0), 2)
            if item.get("change_pct") is None:
                item["change_pct"] = to_float(first_existing(row, ("涨跌幅", "今日涨跌幅", "行业-涨跌幅")))

    for name, count in heat.items():
        matched = False
        for sector_name, item in sectors.items():
            if name in sector_name or sector_name in name:
                item["news_heat"] = max(item.get("news_heat", 0), count)
                matched = True
        if not matched and count > 0:
            sectors[name] = {
                "sector_name": name,
                "change_pct": None,
                "volume_change": None,
                "money_strength": None,
                "news_heat": count,
            }
    return list(sectors.values())


def score_sectors(items: list[dict[str, Any]]) -> list[SectorRank]:
    if not items:
        return [SectorRank(rank=1, sector_name="数据不足", total_score=50, reason="未获取到行业数据", risk_tags=["行业数据不足"])]

    change_values = [item.get("change_pct") for item in items if item.get("change_pct") is not None]
    volume_values = [item.get("volume_change") for item in items if item.get("volume_change") is not None]
    money_values = [item.get("money_strength") for item in items if item.get("money_strength") is not None]
    heat_values = [item.get("news_heat", 0) for item in items]

    ranked: list[SectorRank] = []
    for item in items:
        change_score = normalize_rank_value(item.get("change_pct"), change_values)
        volume_score = normalize_rank_value(item.get("volume_change"), volume_values)
        money_score = normalize_rank_value(item.get("money_strength"), money_values)
        heat_score = normalize_rank_value(item.get("news_heat", 0), heat_values)
        total = round(change_score * 0.35 + volume_score * 0.20 + money_score * 0.30 + heat_score * 0.15, 2)

        risks: list[str] = []
        if item.get("change_pct") is not None and item["change_pct"] > 7:
            risks.append("行业短线涨幅过高")
        if item.get("money_strength") is not None and item["money_strength"] < 0:
            risks.append("主力资金流出")

        reason = f"涨幅分{change_score:.1f}，量能分{volume_score:.1f}，资金分{money_score:.1f}，新闻热度分{heat_score:.1f}"
        ranked.append(
            SectorRank(
                rank=0,
                sector_name=item["sector_name"],
                total_score=total,
                change_pct=item.get("change_pct"),
                volume_change=item.get("volume_change"),
                money_strength=item.get("money_strength"),
                news_heat=int(item.get("news_heat", 0) or 0),
                reason=reason,
                risk_tags=risks,
            )
        )

    ranked.sort(key=lambda sector: sector.total_score, reverse=True)
    for index, sector in enumerate(ranked, start=1):
        sector.rank = index
    return ranked


def write_sector_rank_report(rankings: list[SectorRank], output_file: Path = DEFAULT_REPORT) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 每日行业轮动排名报告",
        "",
        f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 排名",
        "",
        "| 排名 | 行业 | 综合分 | 涨幅 | 量能/换手 | 资金强度 | 新闻热度 | 风险 |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in rankings:
        lines.append(
            "| {rank} | {name} | {score} | {change} | {volume} | {money} | {heat} | {risk} |".format(
                rank=item.rank,
                name=item.sector_name,
                score=item.total_score,
                change=format_value(item.change_pct),
                volume=format_value(item.volume_change),
                money=format_value(item.money_strength),
                heat=item.news_heat,
                risk="、".join(item.risk_tags) if item.risk_tags else "-",
            )
        )

    lines.extend(
        [
            "",
            "## 评分说明",
            "",
            "- 综合分 = 行业涨幅 35% + 行业成交/量能 20% + 行业资金强度 30% + 行业新闻热度 15%。",
            "- 新闻热度来自本地 `logs/trading_log.csv` 中近期新闻标题与行业关键词的匹配。",
            "- 本报告只用于研究和复盘，不产生自动交易。",
        ]
    )
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def calculate_news_heat(log_file: Path = DEFAULT_LOG, max_rows: int = 500) -> dict[str, int]:
    heat = {sector: 0 for sector in INDUSTRY_NEWS_KEYWORDS}
    if not log_file.exists():
        return heat
    try:
        with log_file.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))[-max_rows:]
    except Exception:
        return heat

    for row in rows:
        text = f"{row.get('新闻标题', '')} {row.get('AI逻辑', '')} {row.get('事件类型', '')}"
        for sector, keywords in INDUSTRY_NEWS_KEYWORDS.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                heat[sector] += 1
    return heat


def sector_heat(name: str, heat: dict[str, int]) -> int:
    count = 0
    for sector, value in heat.items():
        if sector in name or name in sector:
            count = max(count, value)
    return count


def normalize_rank_value(value: float | int | None, values: list[float | int]) -> float:
    if value is None or not values:
        return 50.0
    low = min(values)
    high = max(values)
    if high == low:
        return 50.0
    return max(0.0, min(100.0, (float(value) - float(low)) / (float(high) - float(low)) * 100))


def first_existing(row: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row.index:
            value = row[name]
            if value is not None and str(value).strip() and str(value).lower() != "nan":
                return value
    return None


def to_float(value: Any) -> float | None:
    try:
        if value in (None, "", "-"):
            return None
        return float(str(value).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return str(round(value, 2))
    return str(value)


def _call_with_timeout(fn, timeout: int) -> Any:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(queue, fn))
    process.daemon = True
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return f"超过 {timeout} 秒未响应"
    if queue.empty():
        return "无返回"
    status, payload = queue.get()
    return payload if status == "ok" else payload


def _worker(queue: mp.Queue, fn) -> None:
    try:
        queue.put(("ok", fn()))
    except Exception as exc:
        queue.put(("error", str(exc)))


if __name__ == "__main__":
    result = build_sector_rotation_report()
    print(f"已生成 {DEFAULT_REPORT}，共 {len(result)} 个行业")
