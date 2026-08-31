from __future__ import annotations

import datetime as dt
import contextlib
import io
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
HISTORY_CACHE_DIR = BASE_DIR / "data_cache" / "history"
QUOTE_TIMEOUT = float(os.getenv("AI_QUANT_QUOTE_TIMEOUT", "5"))
HISTORY_CACHE_HOURS = int(os.getenv("AI_QUANT_HISTORY_CACHE_HOURS", "18"))


@dataclass
class NewsItem:
    title: str
    content: str
    published_at: str
    source: str = "未知"


@dataclass
class MarketSnapshot:
    code: str
    price: Any = None
    change_pct: Any = None
    volume: Any = None
    source: str = "未知"
    trend: str = "未知"
    summary: str = "未知"
    main_net_in: Any = 0
    pct_change_20d: Any = None
    pct_change_60d: Any = None
    volume_ratio_5_20: Any = None
    gap_pct: Any = None
    turnover_rate: Any = None


def _fetch_news_worker(queue: mp.Queue) -> None:
    sources = [
        ("东方财富", ak.stock_info_global_em),
        ("同花顺", ak.stock_info_global_ths),
        ("新浪", ak.stock_info_global_sina),
        ("财联社", ak.stock_info_global_cls),
    ]
    errors: list[str] = []

    for source_name, fetcher in sources:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                df = fetcher()
            if df.empty:
                errors.append(f"{source_name}: 空数据")
                continue

            row = df.iloc[0]
            title = _first_existing(row, ["标题"])
            content = _first_existing(row, ["内容", "摘要"])
            published_at = _first_existing(row, ["发布时间", "时间"])

            if not title:
                title = content[:40] if content else source_name

            queue.put(
                (
                    "ok",
                    {
                        "title": title,
                        "content": content or title,
                        "published_at": published_at,
                        "source": source_name,
                    },
                )
            )
            return
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")

    queue.put(("error", "所有新闻源都失败：" + " | ".join(errors)))


def _first_existing(row: pd.Series, names: list[str]) -> str:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return str(row[name])
    return ""


def fetch_latest_news(timeout: int) -> NewsItem | None:
    """Fetch the latest CLS news in a child process so slow vendors cannot hang the app."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_fetch_news_worker, args=(queue,))
    process.daemon = True
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"新闻接口超过 {timeout} 秒未响应")

    if queue.empty():
        return None

    status, payload = queue.get()
    if status == "error":
        raise RuntimeError(payload)
    if payload is None:
        return None
    return NewsItem(**payload)


def calculate_kdj(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    low_list = df["最低"].rolling(period, min_periods=period).min()
    high_list = df["最高"].rolling(period, min_periods=period).max()
    rsv = (df["收盘"] - low_list) / (high_list - low_list) * 100
    df["K"] = rsv.ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df


def _call_worker(queue: mp.Queue, call_name: str, code: str) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if call_name == "realtime_price":
                result = _get_realtime_price_direct(code)
            elif call_name == "technical_analysis":
                result = _get_technical_analysis_direct(code)
            elif call_name == "money_flow":
                result = _get_money_flow_direct(code)
            else:
                raise ValueError(f"未知调用: {call_name}")
        queue.put(("ok", result))
    except Exception as exc:
        queue.put(("error", str(exc)))


def _call_with_timeout(call_name: str, code: str, timeout: int) -> dict[str, Any] | None:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_call_worker, args=(queue, call_name, code))
    process.daemon = True
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        print(f"{call_name} 超过 {timeout} 秒未响应，已跳过")
        return None

    if queue.empty():
        return None

    status, payload = queue.get()
    if status == "error":
        print(f"{call_name} 失败: {payload}")
        return None
    return payload


def get_technical_analysis(code: str, timeout: int = 15) -> dict[str, Any] | None:
    return _call_with_timeout("technical_analysis", code, timeout)


def _get_technical_analysis_direct(code: str) -> dict[str, Any] | None:
    end_date = dt.datetime.now().strftime("%Y%m%d")
    start_date = (dt.datetime.now() - dt.timedelta(days=365)).strftime("%Y%m%d")
    df = fetch_daily_history_with_fallback(code, start_date, end_date)
    if df.empty or len(df) < 26:
        return None

    df["MA5"] = df["收盘"].rolling(5).mean()
    df["MA20"] = df["收盘"].rolling(20).mean()
    df["VOL_MA20"] = df["成交量"].rolling(20).mean()
    df["BOLL_MID"] = df["MA20"]
    df["STD"] = df["收盘"].rolling(20).std()
    df["BOLL_UP"] = df["BOLL_MID"] + 2 * df["STD"]
    df["BOLL_LOW"] = df["BOLL_MID"] - 2 * df["STD"]

    ema12 = df["收盘"].ewm(span=12, adjust=False).mean()
    ema26 = df["收盘"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])

    delta = df["收盘"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df = calculate_kdj(df)
    df["VOL_MA5"] = df["成交量"].rolling(5).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    base_20 = df.iloc[-21] if len(df) >= 21 else None
    base_60 = df.iloc[-61] if len(df) >= 61 else None
    signals: list[str] = []

    if latest["MA5"] > latest["MA20"]:
        trend = "均线多头"
        signals.append("均线多头")
    elif latest["MA5"] < latest["MA20"]:
        trend = "均线空头"
        signals.append("均线空头")
    else:
        trend = "均线震荡"

    if latest["DIF"] > latest["DEA"] and prev["DIF"] <= prev["DEA"]:
        signals.append("MACD金叉")
    elif latest["DIF"] < latest["DEA"] and prev["DIF"] >= prev["DEA"]:
        signals.append("MACD死叉")
    elif latest["MACD"] > 0 and latest["MACD"] > prev["MACD"]:
        signals.append("MACD红柱放大")

    if latest["K"] > latest["D"] and prev["K"] <= prev["D"]:
        signals.append("KDJ金叉")
    if latest["RSI"] > 80:
        signals.append("RSI超买风险")
    if latest["RSI"] < 20:
        signals.append("RSI超卖反弹")
    if latest["收盘"] > latest["BOLL_UP"]:
        signals.append("突破布林上轨")
    elif latest["收盘"] < latest["BOLL_LOW"]:
        signals.append("跌破布林下轨")
    if latest["成交量"] > latest["VOL_MA5"] * 1.5:
        signals.append("放量")

    pct_change_20d = None
    if base_20 is not None and base_20["收盘"]:
        pct_change_20d = round((latest["收盘"] - base_20["收盘"]) / base_20["收盘"] * 100, 2)

    pct_change_60d = None
    if base_60 is not None and base_60["收盘"]:
        pct_change_60d = round((latest["收盘"] - base_60["收盘"]) / base_60["收盘"] * 100, 2)

    volume_ratio_5_20 = None
    if latest["VOL_MA20"]:
        volume_ratio_5_20 = round(latest["VOL_MA5"] / latest["VOL_MA20"], 2)

    gap_pct = None
    if prev["收盘"]:
        gap_pct = round((latest["开盘"] - prev["收盘"]) / prev["收盘"] * 100, 2)

    turnover_rate = latest["换手率"] if "换手率" in df.columns else None

    return {
        "date": str(latest["日期"]),
        "close": latest["收盘"],
        "change_pct": round((latest["收盘"] - prev["收盘"]) / prev["收盘"] * 100, 2),
        "trend": trend,
        "summary": " | ".join(signals) if signals else "震荡无明显信号",
        "ma5": round(latest["MA5"], 2),
        "ma20": round(latest["MA20"], 2),
        "rsi": round(latest["RSI"], 1),
        "macd": round(latest["MACD"], 3),
        "kdj_j": round(latest["J"], 1),
        "boll_up": round(latest["BOLL_UP"], 2),
        "pct_change_20d": pct_change_20d,
        "pct_change_60d": pct_change_60d,
        "volume_ratio_5_20": volume_ratio_5_20,
        "gap_pct": gap_pct,
        "turnover_rate": turnover_rate,
    }


def fetch_daily_history_with_fallback(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    errors: list[str] = []
    cached = load_history_cache(code)
    if is_cache_usable(cached, start_date, end_date):
        return trim_history(cached, start_date, end_date)

    df = fetch_eastmoney_history_http(code, start_date, end_date)
    if not df.empty:
        save_history_cache(code, df)
        return df
    errors.append("东方财富HTTP日线: 空数据")

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
        df = normalize_history_columns(df)
        if not df.empty:
            save_history_cache(code, df)
            return df
        errors.append("东方财富日线: 空数据")
    except Exception as exc:
        errors.append(f"东方财富日线: {exc}")

    tx_symbol = market_symbol(code)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = ak.stock_zh_a_hist_tx(symbol=tx_symbol, start_date=start_date, end_date=end_date, adjust="qfq", timeout=6)
        df = normalize_history_columns(df)
        if not df.empty:
            save_history_cache(code, df)
            print("技术分析使用腾讯日线兜底")
            return df
        errors.append("腾讯日线: 空数据")
    except Exception as exc:
        errors.append(f"腾讯日线: {exc}")

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = ak.stock_zh_a_daily(symbol=tx_symbol, start_date=start_date, end_date=end_date, adjust="qfq")
        df = normalize_history_columns(df)
        if not df.empty:
            save_history_cache(code, df)
            print("技术分析使用备用日线接口兜底")
            return df
        errors.append("备用日线: 空数据")
    except Exception as exc:
        errors.append(f"备用日线: {exc}")

    if cached is not None and not cached.empty:
        print(f"历史行情接口失败，使用本地缓存兜底: {code}")
        return trim_history(cached, start_date, end_date)

    print("历史行情兜底失败: " + " | ".join(errors))
    return pd.DataFrame()


def fetch_eastmoney_history_http(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    secid = eastmoney_secid(code)
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": start_date,
        "end": end_date,
    }
    try:
        response = requests.get(url, params=params, timeout=QUOTE_TIMEOUT, headers=request_headers(), proxies=no_proxy())
        response.raise_for_status()
        payload = response.json()
        klines = (((payload or {}).get("data") or {}).get("klines")) or []
        rows: list[dict[str, Any]] = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 11:
                continue
            rows.append(
                {
                    "日期": parts[0],
                    "开盘": parts[1],
                    "收盘": parts[2],
                    "最高": parts[3],
                    "最低": parts[4],
                    "成交量": parts[5],
                    "成交额": parts[6],
                    "振幅": parts[7],
                    "涨跌幅": parts[8],
                    "涨跌额": parts[9],
                    "换手率": parts[10],
                }
            )
        return normalize_history_columns(pd.DataFrame(rows))
    except Exception:
        return pd.DataFrame()


def load_history_cache(code: str) -> pd.DataFrame | None:
    path = history_cache_path(code)
    if not path.exists():
        return None
    try:
        return normalize_history_columns(pd.read_csv(path, encoding="utf-8-sig"))
    except Exception:
        return None


def save_history_cache(code: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    try:
        HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        normalize_history_columns(df).to_csv(history_cache_path(code), index=False, encoding="utf-8-sig")
    except Exception:
        pass


def history_cache_path(code: str) -> Path:
    return HISTORY_CACHE_DIR / f"{code}.csv"


def is_cache_usable(df: pd.DataFrame | None, start_date: str, end_date: str) -> bool:
    if df is None or df.empty:
        return False
    trimmed = trim_history(df, start_date, end_date)
    if trimmed.empty or len(trimmed) < min(30, len(df)):
        return False
    dates = pd.to_datetime(normalize_history_columns(df)["日期"], errors="coerce").dropna()
    start = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")
    if dates.empty:
        return False
    if pd.notna(start) and dates.min() > start + pd.Timedelta(days=7):
        return False
    if pd.notna(end) and dates.max() < end - pd.Timedelta(days=7):
        return False
    return True


def is_cache_fresh(df: pd.DataFrame) -> bool:
    try:
        latest = pd.to_datetime(df["日期"].iloc[-1]).date()
        today = dt.datetime.now().date()
        return latest >= today - dt.timedelta(days=5)
    except Exception:
        return False


def trim_history(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    normalized = normalize_history_columns(df)
    if normalized.empty:
        return normalized
    start = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")
    if pd.notna(start):
        dates = pd.to_datetime(normalized["日期"], errors="coerce")
        normalized = normalized[dates >= start]
    if pd.notna(end):
        dates = pd.to_datetime(normalized["日期"], errors="coerce")
        normalized = normalized[dates <= end]
    return normalized.reset_index(drop=True)


def normalize_history_columns(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    rename_map = {
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "volume": "成交量",
        "vol": "成交量",
        "amount": "成交额",
        "turnover": "换手率",
    }
    df = df.rename(columns={column: rename_map.get(str(column), column) for column in df.columns})
    required = {"日期", "开盘", "收盘", "最高", "最低", "成交量"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    for column in ("开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["开盘", "收盘", "最高", "最低", "成交量"])
    return df.sort_values("日期").reset_index(drop=True)


def market_symbol(code: str) -> str:
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    return f"{prefix}{code}"


def eastmoney_secid(code: str) -> str:
    market = "1" if code.startswith(("6", "9")) else "0"
    return f"{market}.{code}"


def request_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://quote.eastmoney.com/",
    }


def no_proxy() -> dict[str, str]:
    return {"http": "", "https": ""}


def get_realtime_price(code: str, timeout: int = 8) -> dict[str, Any] | None:
    return _call_with_timeout("realtime_price", code, timeout)


def _get_realtime_price_direct(code: str) -> dict[str, Any] | None:
    price = fetch_eastmoney_realtime_http(code) or fetch_sina_realtime_http(code) or fetch_tencent_realtime_http(code)
    if price:
        return price
    try:
        df = ak.stock_zh_a_spot_em()
        stock = df[df["代码"] == code]
        if not stock.empty:
            row = stock.iloc[0]
            return {
                "price": row["最新价"],
                "change_pct": row["涨跌幅"],
                "volume": row["成交量"],
                "source": "东方财富实时行情",
            }
    except Exception as exc:
        print(f"实时行情失败: {exc}")
    return None


def fetch_eastmoney_realtime_http(code: str) -> dict[str, Any] | None:
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": eastmoney_secid(code),
        "fields": "f43,f47,f57,f58,f170",
    }
    try:
        response = requests.get(url, params=params, timeout=QUOTE_TIMEOUT, headers=request_headers(), proxies=no_proxy())
        response.raise_for_status()
        data = ((response.json() or {}).get("data")) or {}
        price = eastmoney_scaled_number(data.get("f43"))
        if price is None:
            return None
        return {
            "price": price,
            "change_pct": eastmoney_scaled_number(data.get("f170")),
            "volume": data.get("f47"),
            "source": "东方财富HTTP实时行情",
        }
    except Exception:
        return None


def fetch_sina_realtime_http(code: str) -> dict[str, Any] | None:
    symbol = market_symbol(code)
    url = f"https://hq.sinajs.cn/list={symbol}"
    try:
        response = requests.get(url, timeout=QUOTE_TIMEOUT, headers=request_headers(), proxies=no_proxy())
        response.raise_for_status()
        text = response.content.decode("gbk", errors="ignore")
        if '=""' in text:
            return None
        body = text.split('"', 2)[1] if '"' in text else ""
        parts = body.split(",")
        if len(parts) < 9:
            return None
        current = parse_number(parts[3])
        previous = parse_number(parts[2])
        if current is None or current <= 0:
            return None
        change_pct = None
        if previous and previous > 0:
            change_pct = round((current - previous) / previous * 100, 2)
        return {
            "price": current,
            "change_pct": change_pct,
            "volume": parse_number(parts[8]),
            "source": "新浪HTTP实时行情",
        }
    except Exception:
        return None


def fetch_tencent_realtime_http(code: str) -> dict[str, Any] | None:
    symbol = market_symbol(code)
    url = f"https://qt.gtimg.cn/q={symbol}"
    try:
        response = requests.get(url, timeout=QUOTE_TIMEOUT, headers=request_headers(), proxies=no_proxy())
        response.raise_for_status()
        text = response.content.decode("gbk", errors="ignore")
        body = text.split('"', 2)[1] if '"' in text else ""
        parts = body.split("~")
        if len(parts) < 33:
            return None
        current = parse_number(parts[3])
        if current is None or current <= 0:
            return None
        return {
            "price": current,
            "change_pct": parse_number(parts[32]),
            "volume": parse_number(parts[6]),
            "source": "腾讯HTTP实时行情",
        }
    except Exception:
        return None


def eastmoney_scaled_number(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number <= -100000:
        return None
    return round(number / 100, 3)


def parse_number(value: Any) -> float | None:
    try:
        if value in (None, "", "-"):
            return None
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def get_money_flow(code: str, timeout: int = 12) -> dict[str, Any] | None:
    return _call_with_timeout("money_flow", code, timeout)


def _get_money_flow_direct(code: str) -> dict[str, Any] | None:
    try:
        market = "sh" if code.startswith("6") else "sz"
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "main_net_in": row["主力净流入-净额"],
            "super_ratio": row["超大单净流入-净占比"],
        }
    except Exception as exc:
        print(f"资金流向失败: {exc}")
        return None


def build_snapshot(code: str) -> MarketSnapshot:
    snapshot = MarketSnapshot(code=code)

    price = get_realtime_price(code)
    if price:
        snapshot.price = price.get("price")
        snapshot.change_pct = price.get("change_pct")
        snapshot.volume = price.get("volume")
        snapshot.source = price.get("source", snapshot.source)

    tech = get_technical_analysis(code)
    if tech:
        snapshot.trend = tech["trend"]
        snapshot.summary = tech["summary"]
        if snapshot.price is None:
            snapshot.price = tech["close"]
            snapshot.change_pct = tech["change_pct"]
            snapshot.source = "日线收盘兜底"
        snapshot.pct_change_20d = tech.get("pct_change_20d")
        snapshot.pct_change_60d = tech.get("pct_change_60d")
        snapshot.volume_ratio_5_20 = tech.get("volume_ratio_5_20")
        snapshot.gap_pct = tech.get("gap_pct")
        snapshot.turnover_rate = tech.get("turnover_rate")

    flow = get_money_flow(code)
    if flow:
        snapshot.main_net_in = flow.get("main_net_in", 0)

    return snapshot
