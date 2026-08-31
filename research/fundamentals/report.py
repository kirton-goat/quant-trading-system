from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from .fundamental_cache import DEFAULT_FUNDAMENTAL_CACHE


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = BASE_DIR / "fundamental_data_quality_report.md"
FIELDS = (
    "revenue",
    "net_profit",
    "operating_cash_flow",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "share_capital",
)


def generate_data_quality_report(
    cache_dir: Path = DEFAULT_FUNDAMENTAL_CACHE,
    output: Path = DEFAULT_REPORT,
) -> str:
    data = load_all_cached_records(cache_dir)
    lines = [
        "# Point-in-Time Fundamentals 数据质量报告",
        "",
        f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 数据来源",
        "",
        "- 利润表、资产负债表、现金流量表：东方财富历史按报告期报表，经 AKShare 接口获取。",
        "- 实际披露日期：各报表的 `NOTICE_DATE`；缓存字段为 `disclosure_date`。",
        "- 数据更新时间：`UPDATE_DATE`；晚于披露日时标记 `is_revised=true`。",
        "- 本地缓存：`data_cache/fundamentals/<stock_code>.csv`。",
        "",
    ]
    if data.empty:
        lines.extend(["当前缓存为空。正式回测会将所有股票标记为 `missing_fundamental_data`，不会使用50分占位。", ""])
    else:
        report_dates = pd.to_datetime(data["report_period"], errors="coerce")
        disclosure_dates = pd.to_datetime(data["disclosure_date"], errors="coerce")
        start_year = int(report_dates.dt.year.min())
        end_year = int(report_dates.dt.year.max())
        disclosure_coverage = disclosure_dates.notna().mean() * 100
        revised = data.get("is_revised", pd.Series(False, index=data.index)).astype(str).str.lower().isin({"true", "1", "yes"}).sum()
        lines.extend(
            [
                "## 缓存覆盖",
                "",
                f"- 股票数：{data['code'].astype(str).nunique()}",
                f"- 财报记录数：{len(data)}",
                f"- 报告期年份：{start_year} 至 {end_year}",
                f"- 披露日期覆盖率：{disclosure_coverage:.2f}%",
                f"- 标记为修订版的记录：{int(revised)}",
                "",
                "## 指标缺失率",
                "",
                "| 字段 | 缺失率 |",
                "|---|---:|",
            ]
        )
        for field in FIELDS:
            missing = pd.to_numeric(data.get(field), errors="coerce").isna().mean() * 100
            lines.append(f"| {field} | {missing:.2f}% |")
        lines.extend(["", "## 各年份可用股票数量", "", "| 年份 | 股票数 | 记录数 |", "|---|---:|---:|"])
        grouped = data.assign(year=report_dates.dt.year).dropna(subset=["year"]).groupby("year")
        for year, group in grouped:
            lines.append(f"| {int(year)} | {group['code'].astype(str).nunique()} | {len(group)} |")
        lines.append("")
    lines.extend(
        [
            "## Point-in-Time 规则",
            "",
            "- `report_period` 只表示财报所属期间，不决定可见时间。",
            "- `effective_date = disclosure_date = NOTICE_DATE`。只有 `disclosure_date <= as_of_date` 才可进入因子计算。",
            "- 新报告披露前继续使用最后一份已披露报告。同比增长只比较已披露的同报告期上一年数据。",
            "- 正式回测缺少可用基本面时剔除股票，不使用最新指标或中性50分。",
            "- 每个调仓日审计结果写入 `fundamental_future_data_check.log`；发现未来记录即阻断。",
            "",
            "## 真实时点示例",
            "",
            *real_example_lines(cache_dir),
            "",
            "## PE/PB 历史计算方法",
            "",
            "- 市值 = 回测日及以前最后一个不复权收盘价 × 当时最后已披露财报股本；价格优先东方财富，失败时使用BaoStock不复权日线。",
            "- PE = 市值 / 当时可构造的TTM归母净利润。非年报期使用“上一年年报 + 本期累计 - 上年同期累计”。",
            "- PB = 市值 / 当时最后已披露的归母权益。",
            "- 无法构造TTM利润、权益、股本或回测日价格时，PE/PB标记缺失，不读取当前估值。",
            "",
            "## 修订财报处理",
            "",
            "- 接口出现同报告期多行时优先最早披露行。",
            "- 若数据源只返回当前修订后的历史值，保留原披露日并标记 `is_revised=true`、记录 `UPDATE_DATE`。",
            "- 这可以消除披露日期穿越，但不能恢复修订前原始数值，仍存在历史版本修订偏差。",
            "",
            "## 当前限制",
            "",
            "- 缓存覆盖取决于已经预取的股票；首次全市场回测需要较长的数据下载时间。",
            "- 金融行业报表结构不同，毛利率或经营现金流等字段可能天然缺失，系统按明确缺失规则处理。",
            "- ROE为基于当期归母利润和平均归母权益的年化近似值，不等同于交易所披露的加权平均ROE。",
            "- 数据源不是逐次公告原始版本库；需要更高等级审计时，应接入交易所公告附件的版本化财报库。",
            "",
        ]
    )
    valuation_count = len(list((cache_dir / "valuation_prices").glob("*.csv")))
    cached_stock_count = int(data["code"].astype(str).nunique()) if not data.empty else 0
    disclosure_coverage = (
        pd.to_datetime(data["disclosure_date"], errors="coerce").notna().mean() * 100
        if not data.empty else 0.0
    )
    summary_path = BASE_DIR / "backtest_v1_summary.json"
    integrity_status = "not_run"
    future_data_count = "unknown"
    if summary_path.exists():
        try:
            import json

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            integrity_status = str(summary.get("integrity_status") or "incomplete")
            future_data_count = sum(
                int(model.get("future_fundamental_data", 0))
                for model in summary.get("models", [])
            )
        except (OSError, ValueError, TypeError):
            integrity_status = "unreadable"
    lines.extend(
        [
            "## Backtest v1.0 Full-Cache Summary",
            "",
            f"- Cached statement stocks: {cached_stock_count}",
            f"- Cached unadjusted valuation-price stocks: {valuation_count}",
            f"- Disclosure-date coverage: {disclosure_coverage:.2f}%",
            "- Visibility rule: disclosure_date <= as_of_date.",
            "- A missing PE/PB remains missing; current valuation data is never substituted.",
            f"- Latest Backtest v1.0 integrity status: {integrity_status}",
            f"- Future fundamental records used by the latest run: {future_data_count}",
            "",
        ]
    )
    text = "\n".join(lines)
    output.write_text(text, encoding="utf-8")
    return text


def real_example_lines(cache_dir: Path) -> list[str]:
    from .point_in_time_fundamentals import get_fundamentals

    result = get_fundamentals(
        "600519",
        "2020-06-30",
        cache_dir=cache_dir,
        allow_network=False,
    )
    if result is None:
        return ["- 本地尚未缓存600519，运行基本面预取后可生成2020-06-30实例。"]
    return [
        f"- 回测日：2020-06-30；股票：600519 贵州茅台。",
        f"- 系统选中报告期：{result.report_period}；披露日：{result.disclosure_date}。",
        "- 原因：该报告已在回测日前公开；2020年中报披露日在2020-06-30之后，因此当日不可见。",
    ]


def load_all_cached_records(cache_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if cache_dir.exists():
        for path in cache_dir.glob("*.csv"):
            try:
                frames.append(pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig"))
            except (OSError, pd.errors.ParserError):
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成历史基本面数据质量报告")
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_data_quality_report(output=Path(args.output))
    print(f"基本面数据质量报告已生成: {args.output}")
