from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from research.factor_test import (
    DEFAULT_LOG,
    FactorObservation,
    FactorTestResult,
    load_stock_codes_from_log,
    parse_codes,
    run_factor_tests,
)
from research.ic_analysis import ICResult, calculate_ic
from stock_pool import DEFAULT_STOCK_POOL_FILE, load_stock_pool


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = BASE_DIR / "factor_research_report.md"


def generate_factor_research_report(
    stock_codes: list[str] | None = None,
    output_file: Path = DEFAULT_OUTPUT,
    pool_file: Path = DEFAULT_STOCK_POOL_FILE,
    holding_days: int = 20,
    history_days: int = 260,
    sample_step: int = 5,
) -> tuple[list[FactorTestResult], list[ICResult], list[FactorObservation]]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    codes = stock_codes or load_stock_pool(pool_file) or load_stock_codes_from_log(DEFAULT_LOG)
    if not codes:
        output_file.write_text(build_no_data_report(), encoding="utf-8")
        return [], [], []

    factor_results, observations = run_factor_tests(
        codes,
        holding_days=holding_days,
        history_days=history_days,
        sample_step=sample_step,
    )
    ic_results = calculate_ic(observations)
    output_file.write_text(
        build_report(codes, factor_results, ic_results, observations, holding_days, history_days, sample_step),
        encoding="utf-8",
    )
    return factor_results, ic_results, observations


def build_no_data_report() -> str:
    return "\n".join(
        [
            "# 因子研究报告",
            "",
            f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 结论",
            "",
            "当前 `logs/trading_log.csv` 中没有可用于因子研究的 6 位 A 股代码。",
            "",
            "请使用以下命令手动传入股票池：",
            "",
            "```powershell",
            'python -u -B -m research.factor_report --codes "601127,600733,300750"',
            "python -u -B -m research.factor_report --pool-file stock_pool.csv",
            "```",
            "",
            "本报告只用于研究，不修改交易逻辑，不产生自动交易。",
            "",
        ]
    )


def build_report(
    codes: list[str],
    factor_results: list[FactorTestResult],
    ic_results: list[ICResult],
    observations: list[FactorObservation],
    holding_days: int,
    history_days: int,
    sample_step: int,
) -> str:
    lines = [
        "# 因子研究报告",
        "",
        f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 研究范围",
        "",
        f"- 股票池数量：{len(codes)}",
        f"- 股票池：{', '.join(codes[:50])}{' ...' if len(codes) > 50 else ''}",
        f"- 未来收益观察窗口：T+{holding_days} 日",
        f"- 历史数据长度：约 {history_days} 个交易日",
        f"- 抽样步长：每 {sample_step} 个交易日取一个样本",
        f"- 总观测样本：{len(observations)}",
        "",
        "## 因子分层收益",
        "",
        "| 因子 | 样本数 | 平均未来收益 | 高分组收益 | 低分组收益 | 多空差 | 胜率 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in factor_results:
        lines.append(
            "| {name} | {obs} | {avg} | {top} | {bottom} | {spread} | {win} | {note} |".format(
                name=result.factor_name,
                obs=result.observations,
                avg=format_pct(result.average_forward_return),
                top=format_pct(result.top_quantile_return),
                bottom=format_pct(result.bottom_quantile_return),
                spread=format_pct(result.spread_return),
                win=format_ratio(result.win_rate),
                note=result.note,
            )
        )

    lines.extend(
        [
            "",
            "## IC 分析",
            "",
            "| 因子 | 样本数 | Pearson IC | Spearman IC | 方向 | 说明 |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for result in ic_results:
        lines.append(
            "| {name} | {obs} | {pearson} | {spearman} | {direction} | {note} |".format(
                name=result.factor_name,
                obs=result.observations,
                pearson=format_number(result.pearson_ic),
                spearman=format_number(result.spearman_ic),
                direction=result.ic_direction,
                note=result.note,
            )
        )

    lines.extend(
        [
            "",
            "## 初步判断",
            "",
            build_summary(factor_results, ic_results),
            "",
            "## 下一步",
            "",
            "1. 扩大股票池，优先使用沪深300、中证500或自定义行业股票池。",
            "2. 固定样本区间后，分别测试动量、资金、基本面因子的单因子表现。",
            "3. 建立组合回测：先测试不含 AI 事件因子的基础模型，再测试加入 AI 事件因子后的增量价值。",
            "4. 加入因子相关性检查，避免多个高度相似因子重复计权。",
            "5. 在回测和风控稳定前，不接入真实交易。",
            "",
            "## 说明",
            "",
            "- 本报告是因子研究，不是交易建议。",
            "- 当前基本面因子为研究框架占位，历史基本面数据尚未完整接入，因此默认偏中性。",
            "- 免费行情接口可能因网络节点异常导致样本不足，报告应结合样本数判断可靠性。",
            "",
        ]
    )
    return "\n".join(lines)


def build_summary(factor_results: list[FactorTestResult], ic_results: list[ICResult]) -> str:
    positives = [item.factor_name for item in factor_results if item.spread_return is not None and item.spread_return > 0]
    ic_positive = [item.factor_name for item in ic_results if item.spearman_ic is not None and item.spearman_ic > 0.03]
    parts: list[str] = []
    if positives:
        parts.append(f"高分组收益暂时优于低分组的因子：{', '.join(positives)}。")
    else:
        parts.append("当前样本中尚未看到明确的高分组收益优势。")
    if ic_positive:
        parts.append(f"Spearman IC 有弱正相关迹象的因子：{', '.join(ic_positive)}。")
    else:
        parts.append("当前样本中 IC 暂未体现稳定预测能力。")
    parts.append("由于股票池和样本期较小，本结论只用于检查研究框架是否跑通。")
    return "\n\n".join(parts)


def format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{round(value, 4)}%"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{round(value * 100, 2)}%"


def format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return str(round(value, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="因子研究报告生成器")
    parser.add_argument("--codes", help="逗号分隔股票池，例如 601127,600733,300750")
    parser.add_argument("--pool-file", default=str(DEFAULT_STOCK_POOL_FILE), help="股票池CSV路径")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出报告路径")
    parser.add_argument("--holding-days", type=int, default=20, help="未来收益观察窗口")
    parser.add_argument("--history-days", type=int, default=260, help="历史行情长度")
    parser.add_argument("--sample-step", type=int, default=5, help="抽样步长")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    codes = parse_codes(args.codes) or load_stock_pool(Path(args.pool_file)) or load_stock_codes_from_log(DEFAULT_LOG)
    factor_results, ic_results, observations = generate_factor_research_report(
        stock_codes=codes,
        output_file=Path(args.output),
        pool_file=Path(args.pool_file),
        holding_days=args.holding_days,
        history_days=args.history_days,
        sample_step=args.sample_step,
    )
    print(f"因子研究报告已生成: {args.output}")
    print(f"股票数: {len(codes)}，观测样本: {len(observations)}")
