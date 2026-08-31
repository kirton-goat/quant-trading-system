from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from ai_engine import AiEngine
from backtest_engine import backtest_log_file
from config import load_settings
from event_research_data import record_research_event
from event_source_engine import fetch_latest_event
from market import MarketSnapshot, NewsItem, build_snapshot
from multi_factor_model import MultiFactorDecision, MultiFactorModel
from news_quality_filter import NewsQualityFilter, NewsQualityReport, combine_scores
from simulation_engine import build_simulated_trade_plan
from research.factor_report import DEFAULT_OUTPUT as DEFAULT_FACTOR_REPORT_OUTPUT
from research.factor_report import generate_factor_research_report
from research.portfolio_backtest import DEFAULT_OUTPUT as DEFAULT_PORTFOLIO_BACKTEST_OUTPUT
from research.portfolio_backtest import generate_portfolio_backtest_report
from research.a_share_backtest import DEFAULT_OUTPUT as DEFAULT_A_SHARE_BACKTEST_OUTPUT
from research.a_share_backtest import generate_a_share_backtest_report
from stock_ranker import StockRanker
from stock_pool import DEFAULT_STOCK_POOL_FILE, load_stock_pool
from storage import append_log


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["NO_PROXY"] = "*"


def print_header() -> None:
    print("\n" + "=" * 64)
    print("AI量化研究系统")
    print("主流程：股票池 -> 市场环境 -> 多因子评分 -> 风险过滤 -> 模拟交易/回测")
    print("信息层：公告/政策可作辅助研究；普通财经新闻仅观察与记录")
    print("当前事件源：公告/政策/东方财富/同花顺/新浪/财联社 | 行情：东方财富+历史兜底 | AI：DeepSeek")
    print("=" * 64)


def display_result(
    news: NewsItem,
    code: str | None,
    snapshot: MarketSnapshot | None,
    decision: dict,
    quality: NewsQualityReport | None = None,
    combined_score: float | None = None,
    multi_factor: MultiFactorDecision | None = None,
    event_type: str = "财经新闻",
) -> None:
    print(f"\n事件：[{event_type}] [{news.source}] {news.published_at} | {news.title}")
    print(f"识别结果：{code or '未识别具体A股/概念'}")
    if quality:
        print(
            "新闻质量："
            f"综合={quality.quality_score:.2f}，交易价值={quality.trade_value_score:.2f}，"
            f"来源L{quality.source_level}/{quality.source_score:.2f}，新颖性={quality.novelty_score:.2f}，"
            f"基本面={quality.fundamental_score:.2f}，市场反应={quality.market_reaction_score:.2f}，"
            f"类型={quality.information_type}"
        )
        print(
            "真实性字段："
            f"is_original={quality.is_original}，is_repeat={quality.is_repeat}，"
            f"has_financial_impact={quality.has_financial_impact}，"
            f"market_already_priced={quality.market_already_priced}，trade_value={quality.trade_value}"
        )
        if quality.risk:
            print(f"风险标签：{' | '.join(quality.risk)}")
    if snapshot:
        print(
            "行情："
            f"价格={snapshot.price}，涨跌幅={snapshot.change_pct}，"
            f"趋势={snapshot.trend}，摘要={snapshot.summary}，来源={snapshot.source}"
        )
        print(
            "提前反应指标："
            f"20日涨幅={snapshot.pct_change_20d}，"
            f"60日涨幅={snapshot.pct_change_60d}，"
            f"5/20量能比={snapshot.volume_ratio_5_20}，"
            f"跳空幅度={snapshot.gap_pct}，"
            f"换手率={snapshot.turnover_rate}"
        )
    print(f"AI研究：评分={decision.get('score')}，建议={decision.get('action')}，逻辑={decision.get('logic')}")
    if multi_factor:
        print(
            "多因子："
            f"总分={multi_factor.total_score}，"
            f"事件={multi_factor.factor_score('event'):.1f}，"
            f"市场={multi_factor.factor_score('market_regime'):.1f}，"
            f"许可={multi_factor.allowed_to_trade}，原因={multi_factor.reason}"
        )
        if multi_factor.risk_tags:
            print(f"多因子风险：{' | '.join(multi_factor.risk_tags)}")
    if combined_score is not None:
        print(f"研究评分：{combined_score}（仅用于事件记录，不是交易评分）")


def process_once(engine: AiEngine) -> None:
    settings = load_settings()
    event = fetch_latest_event(timeout=settings.news_timeout)
    if event is None:
        print("没有获取到事件。")
        return

    news = event.news
    code = event.forced_code or engine.extract_stock_code(news)
    snapshot = build_snapshot(code) if is_stock_code(code) else None
    quality_filter = NewsQualityFilter(settings.log_file)
    quality = quality_filter.evaluate(news, mapped_target=code, snapshot=snapshot)
    decision = engine.decide(news, code, snapshot)
    decision = apply_quality_threshold(decision, quality)
    multi_factor = evaluate_multi_factor(code, news, snapshot, quality, event.event_type)
    decision = apply_multi_factor_gate(decision, code, multi_factor)
    combined_score = combine_scores(quality.quality_score, decision.get("score"), quality.technical_score)
    sim_plan = build_simulated_trade_plan(news, code, snapshot, "观望")

    display_result(news, code, snapshot, decision, quality, combined_score, multi_factor, event.event_type)
    record_research_event(
        event.event_type,
        news.published_at,
        code,
        quality.trade_value_score * 100,
        news.source,
        news.title,
        news.content,
    )

    append_log(
        settings.log_file,
        {
            "时间": news.published_at,
            "新闻来源": news.source,
            "事件类型": event.event_type,
            "新闻标题": news.title,
            "关联股票": code or "无",
            "当前价格": snapshot.price if snapshot else "-",
            "涨跌幅": snapshot.change_pct if snapshot else "-",
            "行情来源": snapshot.source if snapshot else "-",
            "技术趋势": snapshot.trend if snapshot else "-",
            "技术摘要": snapshot.summary if snapshot else "-",
            "20日涨幅": snapshot.pct_change_20d if snapshot else "-",
            "60日涨幅": snapshot.pct_change_60d if snapshot else "-",
            "5/20量能比": snapshot.volume_ratio_5_20 if snapshot else "-",
            "跳空幅度": snapshot.gap_pct if snapshot else "-",
            "换手率": snapshot.turnover_rate if snapshot else "-",
            "information_type": quality.information_type,
            "source_level": quality.source_level,
            "source_score": round(quality.source_score * 100, 0),
            "is_original": quality.is_original,
            "is_repeat": quality.is_repeat,
            "has_financial_impact": quality.has_financial_impact,
            "market_already_priced": quality.market_already_priced,
            "trade_value": quality.trade_value,
            "新闻质量评分": round(quality.quality_score, 2),
            "新闻交易价值评分": round(quality.trade_value_score, 2),
            "技术指标评分": round(quality.technical_score, 2),
            "AI评分": decision.get("score"),
            "AI原始操作": decision.get("raw_action", decision.get("action")),
            "质量阈值拦截": decision.get("quality_blocked", False),
            **multi_factor_log_fields(multi_factor),
            "综合评分": combined_score,
            "AI操作": decision.get("action"),
            "AI逻辑": decision.get("logic"),
            "风险标签": " | ".join(quality.risk),
            **sim_plan.to_log_fields(),
        },
    )

    print("事件已记录为研究信息，未生成模拟开仓或交易推送。")


def make_manual_news(title: str, content: str) -> NewsItem:
    return NewsItem(title=title, content=content, published_at="手动输入", source="手动")


def is_stock_code(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"\d{6}", value))


def apply_quality_threshold(decision: dict, quality: NewsQualityReport, threshold: float = 0.45) -> dict:
    result = dict(decision)
    raw_action = result.get("action", "观望")
    result["raw_action"] = raw_action
    result["quality_blocked"] = False
    if quality.quality_score < threshold and raw_action != "观望":
        result["action"] = "观望"
        result["quality_blocked"] = True
        result["logic"] = (
            f"新闻质量评分{quality.quality_score:.2f}低于{threshold:.2f}，"
            f"原建议{raw_action}被降级为观望；"
            f"{result.get('logic', '')}"
        )[:180]
    return result


def evaluate_multi_factor(
    code: str | None,
    news: NewsItem,
    snapshot: MarketSnapshot | None,
    quality: NewsQualityReport,
    event_type: str = "财经新闻",
) -> MultiFactorDecision | None:
    if not is_stock_code(code):
        return None
    context = {
        "news": news,
        "news_text": f"{news.title} {news.content}",
        "snapshot": snapshot,
        "news_quality": quality,
        "event_type": event_type,
    }
    try:
        return MultiFactorModel().evaluate(code, context=context)
    except Exception as exc:
        print(f"多因子评估失败: {exc}")
        return None


def apply_multi_factor_gate(
    decision: dict,
    code: str | None,
    multi_factor: MultiFactorDecision | None,
) -> dict:
    result = dict(decision)
    raw_action = result.get("action", "观望")
    result["raw_action"] = result.get("raw_action", raw_action)
    result["action"] = "观望"
    result["event_trade_disabled"] = True
    result["logic"] = (
        f"事件扫描不生成交易信号，原AI建议{raw_action}仅作研究记录；"
        "实际模拟交易只能由股票池横向多因子排名、市场环境和风险过滤共同产生。"
    )[:180]
    return result


def multi_factor_log_fields(multi_factor: MultiFactorDecision | None) -> dict:
    if multi_factor is None:
        return {
            "多因子综合评分": "-",
            "动量因子": "-",
            "资金因子": "-",
            "事件因子": "-",
            "技术因子": "-",
            "基本面因子": "-",
            "市场环境因子": "-",
            "多因子交易许可": False,
            "多因子拦截原因": "未评估或无具体股票",
        }
    return {
        "多因子综合评分": multi_factor.total_score,
        "动量因子": round(multi_factor.factor_score("momentum"), 2),
        "资金因子": round(multi_factor.factor_score("money_flow"), 2),
        "事件因子": round(multi_factor.factor_score("event"), 2),
        "技术因子": round(multi_factor.factor_score("technical"), 2),
        "基本面因子": round(multi_factor.factor_score("fundamental"), 2),
        "市场环境因子": round(multi_factor.factor_score("market_regime"), 2),
        "多因子交易许可": multi_factor.allowed_to_trade,
        "多因子拦截原因": multi_factor.reason,
    }


def handle_news(
    engine: AiEngine,
    news: NewsItem,
    forced_code: str | None = None,
    quality_filter: NewsQualityFilter | None = None,
    event_type: str = "财经新闻",
) -> None:
    settings = load_settings()
    code = forced_code or engine.extract_stock_code(news)
    snapshot = build_snapshot(code) if is_stock_code(code) else None
    quality_filter = quality_filter or NewsQualityFilter(settings.log_file)
    quality = quality_filter.evaluate(news, mapped_target=code, snapshot=snapshot)
    decision = engine.decide(news, code, snapshot)
    decision = apply_quality_threshold(decision, quality)
    multi_factor = evaluate_multi_factor(code, news, snapshot, quality, event_type)
    decision = apply_multi_factor_gate(decision, code, multi_factor)
    combined_score = combine_scores(quality.quality_score, decision.get("score"), quality.technical_score)
    sim_plan = build_simulated_trade_plan(news, code, snapshot, "观望")
    display_result(news, code, snapshot, decision, quality, combined_score, multi_factor, event_type)
    record_research_event(
        event_type,
        news.published_at,
        code,
        quality.trade_value_score * 100,
        news.source,
        news.title,
        news.content,
    )

    append_log(
        settings.log_file,
        {
            "时间": news.published_at,
            "新闻来源": news.source,
            "事件类型": event_type,
            "新闻标题": news.title,
            "关联股票": code or "无",
            "当前价格": snapshot.price if snapshot else "-",
            "涨跌幅": snapshot.change_pct if snapshot else "-",
            "行情来源": snapshot.source if snapshot else "-",
            "技术趋势": snapshot.trend if snapshot else "-",
            "技术摘要": snapshot.summary if snapshot else "-",
            "20日涨幅": snapshot.pct_change_20d if snapshot else "-",
            "60日涨幅": snapshot.pct_change_60d if snapshot else "-",
            "5/20量能比": snapshot.volume_ratio_5_20 if snapshot else "-",
            "跳空幅度": snapshot.gap_pct if snapshot else "-",
            "换手率": snapshot.turnover_rate if snapshot else "-",
            "information_type": quality.information_type,
            "source_level": quality.source_level,
            "source_score": round(quality.source_score * 100, 0),
            "is_original": quality.is_original,
            "is_repeat": quality.is_repeat,
            "has_financial_impact": quality.has_financial_impact,
            "market_already_priced": quality.market_already_priced,
            "trade_value": quality.trade_value,
            "新闻质量评分": round(quality.quality_score, 2),
            "新闻交易价值评分": round(quality.trade_value_score, 2),
            "技术指标评分": round(quality.technical_score, 2),
            "AI评分": decision.get("score"),
            "AI原始操作": decision.get("raw_action", decision.get("action")),
            "质量阈值拦截": decision.get("quality_blocked", False),
            **multi_factor_log_fields(multi_factor),
            "综合评分": combined_score,
            "AI操作": decision.get("action"),
            "AI逻辑": decision.get("logic"),
            "风险标签": " | ".join(quality.risk),
            **sim_plan.to_log_fields(),
        },
    )

    print("事件已记录为研究信息，未生成模拟开仓或交易推送。")


def run_loop(
    once: bool,
    pause_on_exit: bool,
    manual_title: str | None = None,
    manual_content: str | None = None,
    forced_code: str | None = None,
) -> None:
    settings = load_settings()
    engine = AiEngine(settings)
    quality_filter = NewsQualityFilter(settings.log_file)

    print_header()
    if not settings.deepseek_api_key:
        print("提示：未设置 DEEPSEEK_API_KEY，AI识别和决策会使用保守占位结果。")

    if manual_title or manual_content:
        news = make_manual_news(manual_title or "手动新闻", manual_content or manual_title or "")
        handle_news(engine, news, forced_code=forced_code, quality_filter=quality_filter, event_type="手动输入")
        if pause_on_exit:
            input("\n按回车键退出...")
        return

    last_title = ""
    seen_titles: set[str] = set()
    while True:
        try:
            event = fetch_latest_event(timeout=settings.news_timeout, exclude_titles=seen_titles)
            if event is None:
                print("没有获取到新事件。")
            elif event.news.title != last_title:
                handle_news(
                    engine,
                    event.news,
                    forced_code=forced_code or event.forced_code,
                    quality_filter=quality_filter,
                    event_type=event.event_type,
                )
                seen_titles.add(event.news.title)
                if len(seen_titles) > 200:
                    seen_titles = set(list(seen_titles)[-100:])
                last_title = event.news.title
            else:
                print(".", end="", flush=True)

            if once:
                print("\n单轮检查完成。")
                break
            time.sleep(settings.interval_seconds)
        except KeyboardInterrupt:
            print("\n已停止。")
            break
        except Exception as exc:
            print(f"\n运行故障：{exc}")
            if once:
                break
            time.sleep(settings.interval_seconds)

    if pause_on_exit:
        input("\n按回车键退出...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI量化新闻交易系统")
    parser.add_argument("--once", action="store_true", help="只运行一轮")
    parser.add_argument("--pause-on-exit", action="store_true", help="结束前等待回车")
    parser.add_argument("--title", help="手动输入新闻标题，用于测试")
    parser.add_argument("--content", help="手动输入新闻内容，用于测试")
    parser.add_argument("--code", help="手动指定6位A股代码，跳过AI识别")
    parser.add_argument("--backtest-log", help="研究模式：读取指定日志文件并回测模拟交易字段")
    parser.add_argument("--holding-days", type=int, default=3, help="模拟交易/回测默认持有天数")
    parser.add_argument("--rank-stocks", help="研究模式：逗号分隔股票池，例如 601127,600733")
    parser.add_argument("--stock-pool", default=str(DEFAULT_STOCK_POOL_FILE), help="研究模式：股票池CSV路径")
    parser.add_argument("--factor-report", action="store_true", help="研究模式：生成因子研究报告")
    parser.add_argument("--portfolio-backtest", action="store_true", help="研究模式：运行基础组合回测")
    parser.add_argument("--a-share-backtest", action="store_true", help="研究模式：运行A股多股票池组合回测")
    parser.add_argument("--universe", default="hs300_csi500", help="A股研究股票池：hs300/csi500/hs300_csi500/all_a/manual")
    parser.add_argument("--universe-limit", type=int, help="A股研究股票池数量上限，用于快速测试")
    parser.add_argument("--research-codes", help="研究模式：逗号分隔股票池，优先于 --stock-pool")
    parser.add_argument("--research-output", default=str(DEFAULT_FACTOR_REPORT_OUTPUT), help="研究报告输出路径")
    parser.add_argument("--portfolio-output", default=str(DEFAULT_PORTFOLIO_BACKTEST_OUTPUT), help="组合回测报告输出路径")
    parser.add_argument("--a-share-output", default=str(DEFAULT_A_SHARE_BACKTEST_OUTPUT), help="A股多股票池回测报告输出路径")
    parser.add_argument("--research-history-days", type=int, default=260, help="因子研究历史行情长度")
    parser.add_argument("--sample-step", type=int, default=5, help="因子研究抽样步长")
    parser.add_argument("--rebalance-days", type=int, default=20, help="组合回测调仓间隔")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="组合回测初始资金")
    parser.add_argument("--fee-rate", type=float, default=0.0015, help="组合回测单边费率")
    parser.add_argument("--top-n", type=int, default=20, help="股票排名输出数量")
    return parser.parse_args()


def run_rank_stocks(stock_text: str, top_n: int, pool_file: str | None = None) -> None:
    codes = [item.strip() for item in re.split(r"[,，\\s]+", stock_text or "") if item.strip()]
    if not codes and pool_file:
        codes = load_stock_pool(Path(pool_file))
    ranked = StockRanker().rank(codes, top_n=top_n)
    if not ranked:
        print("没有可排名的6位A股代码。")
        return
    print("\n多因子股票排名")
    print("排名 | 股票 | 总分 | 动量 | 资金 | 事件 | 技术 | 基本面 | 市场 | 许可 | 原因")
    for item in ranked:
        scores = item.factor_scores
        print(
            f"{item.rank} | {item.code} | {item.total_score} | "
            f"{scores.get('momentum', 0)} | {scores.get('money_flow', 0)} | "
            f"{scores.get('event', 0)} | {scores.get('technical', 0)} | "
            f"{scores.get('fundamental', 0)} | {scores.get('market_regime', 0)} | "
            f"{item.allowed_to_trade} | {item.reason}"
        )
        if item.risk_tags:
            print(f"  风险：{' | '.join(item.risk_tags)}")


if __name__ == "__main__":
    args = parse_args()
    if args.backtest_log:
        summary = backtest_log_file(Path(args.backtest_log), holding_days=args.holding_days)
        print(summary)
        raise SystemExit(0)
    if args.factor_report:
        research_codes = [item.strip() for item in re.split(r"[,，\\s]+", args.research_codes or "") if item.strip()]
        factor_results, ic_results, observations = generate_factor_research_report(
            stock_codes=research_codes or None,
            output_file=Path(args.research_output),
            pool_file=Path(args.stock_pool),
            holding_days=args.holding_days,
            history_days=args.research_history_days,
            sample_step=args.sample_step,
        )
        print(f"因子研究报告已生成: {args.research_output}")
        print(f"因子数量: {len(factor_results)}，IC数量: {len(ic_results)}，观测样本: {len(observations)}")
        raise SystemExit(0)
    if args.portfolio_backtest:
        research_codes = [item.strip() for item in re.split(r"[,，\\s]+", args.research_codes or "") if item.strip()]
        codes = research_codes or None
        result = generate_portfolio_backtest_report(
            stock_codes=codes,
            output_file=Path(args.portfolio_output),
            pool_file=Path(args.stock_pool),
            top_n=args.top_n,
            holding_days=args.holding_days,
            rebalance_days=args.rebalance_days,
            history_days=args.research_history_days,
            initial_capital=args.initial_capital,
            fee_rate=args.fee_rate,
            universe_type=args.universe,
            universe_limit=args.universe_limit,
        )
        print(f"基础组合回测报告已生成: {args.portfolio_output}")
        print(f"调仓次数: {result.trade_count}，总收益率: {result.total_return_pct}%")
        raise SystemExit(0)
    if args.a_share_backtest:
        result = generate_a_share_backtest_report(
            output_file=Path(args.a_share_output),
            universe_name=args.universe,
            universe_limit=args.universe_limit,
            top_n=args.top_n,
            holding_days=args.holding_days,
            rebalance_days=args.rebalance_days,
            history_days=args.research_history_days,
            initial_capital=args.initial_capital,
            fee_rate=args.fee_rate,
        )
        print(f"A股多股票池回测报告已生成: {args.a_share_output}")
        print(
            f"股票数: {result.available_stock_count}/{result.requested_stock_count}，"
            f"调仓次数: {len(result.trades)}，总收益率: {result.total_return_pct}%"
        )
        raise SystemExit(0)
    if args.rank_stocks is not None:
        run_rank_stocks(args.rank_stocks, args.top_n, pool_file=args.stock_pool)
        raise SystemExit(0)
    run_loop(
        once=args.once,
        pause_on_exit=args.pause_on_exit,
        manual_title=args.title,
        manual_content=args.content,
        forced_code=args.code,
    )
