from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from research.universe.historical_universe import resolve_historical_universe
from research.universe.stock_filter import UniverseFilterConfig
from universe_manager import load_universe_cache


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = BASE_DIR / "historical_universe_report.md"
DEFAULT_DATES = ["2018-06-29", "2020-06-30", "2025-06-30"]


def generate_report(
    dates: list[str] | None = None,
    universe_type: str = "CSI300_CSI500",
    output_file: Path = DEFAULT_OUTPUT,
    allow_network: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date in dates or DEFAULT_DATES:
        result = resolve_historical_universe(
            date,
            universe_type,
            filter_config=UniverseFilterConfig(require_market_history=False),
            allow_network=allow_network,
        )
        stats = result.filter_stats
        rows.append(
            {
                "date": result.date,
                "raw_count": result.raw_count,
                "filtered_count": len(result.stocks),
                "st": stats.st,
                "suspended": stats.suspended,
                "delisted": stats.delisted,
                "future_listing": stats.future_listing,
                "future_membership": stats.future_membership,
                "illiquid": stats.illiquid,
                "snapshot_dates": ", ".join(result.snapshot_dates),
                "source": result.source,
                "codes": set(result.codes),
            }
        )
    output_file.write_text(build_report(rows, universe_type), encoding="utf-8")
    return rows


def build_report(rows: list[dict[str, Any]], universe_type: str) -> str:
    current_codes = {
        item.stock_code
        for kind in ("hs300", "csi500")
        for item in load_universe_cache(kind)
    }
    lines = [
        "# Historical Universe 历史股票池报告",
        "",
        f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 结论",
        "",
        "`stock_pool.csv` 仅用于显式开发测试，不再是正式组合回测的默认股票池。沪深300/中证500回测现在按每个调仓日读取当时真实成分；缺少历史快照时直接停止，不会使用当前成分股回填过去。",
        "",
        "## 数据来源",
        "",
        "- 历史指数成分：BaoStock `query_hs300_stocks(date=...)`、`query_zz500_stocks(date=...)`。",
        "- 历史交易状态：BaoStock `query_all_stock(day=...)`。",
        "- 上市/退市日期：BaoStock `query_stock_basic()`。",
        "- 本地快照目录：`data_cache/historical_universe/index_members/`。",
        "- 中证指数当前成分接口只用于当前研究展示，不允许作为历史回测兜底。",
        "",
        "## 日期样本",
        "",
        f"股票池类型：{universe_type}",
        "",
        "| 回测日期 | 快照生效日 | 原始成分 | 静态过滤后 | ST | 当日停牌 | 已退市 | 未来上市/未来成分 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['date']} | {row['snapshot_dates']} | {row['raw_count']} | {row['filtered_count']} | "
            f"{row['st']} | {row['suspended']} | {row['delisted']} | {row['future_listing'] + row['future_membership']} |"
        )
    lines.extend(
        [
            "",
            "说明：报告阶段执行可由历史状态直接验证的过滤。上市满180个交易日、长期停牌和过去20日平均成交额过滤，在正式回测中使用截至调仓日的历史行情逐期执行；当前本地行情缓存尚未覆盖全部早期股票，因此本报告不伪造流动性统计。",
            "",
            "## 当前名单与历史名单差异",
            "",
        ]
    )
    if current_codes:
        for row in rows:
            added_later = current_codes - row["codes"]
            historical_only = row["codes"] - current_codes
            lines.append(
                f"- {row['date']}：当前名单中有 {len(added_later)} 只不在当时过滤后股票池；当时股票池有 {len(historical_only)} 只已不在当前名单。"
            )
    else:
        lines.append("- 当前成分股缓存不可用，未计算集合差异。")
    lines.extend(
        [
            "",
            "## 防未来数据规则",
            "",
            "- 快照生效日晚于回测日：阻断。",
            "- 上市日期晚于回测日：阻断。",
            "- 退市日期早于回测日：过滤。",
            "- 指定历史日期没有快照：报错停止，不读取最新名单。",
            "- 审计日志：`future_data_check.log`。",
            "",
            "## 当前数据限制",
            "",
            "历史成分快照已经可用，但严格长周期组合回测还要求相同区间的完整个股行情。若行情缓存不足，系统会减少候选或停止该期调仓，不会降低过滤阈值来制造结果。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成历史股票池验证报告")
    parser.add_argument("--dates", default=",".join(DEFAULT_DATES), help="逗号分隔历史日期")
    parser.add_argument("--universe", default="CSI300_CSI500")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--offline", action="store_true", help="只使用本地历史快照")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dates = [item.strip() for item in args.dates.split(",") if item.strip()]
    results = generate_report(dates, args.universe, Path(args.output), allow_network=not args.offline)
    print(f"历史股票池报告已生成: {args.output}")
    print("; ".join(f"{row['date']}={row['filtered_count']}只" for row in results))
