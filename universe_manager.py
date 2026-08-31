from __future__ import annotations

import csv
import datetime as dt
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stock_pool import load_stock_pool_items


BASE_DIR = Path(__file__).resolve().parent
UNIVERSE_CACHE_DIR = BASE_DIR / "data_cache" / "universe"
DEFAULT_UNIVERSE = "hs300_csi500"

INDEX_SYMBOLS = {
    "hs300": ("000300", "沪深300"),
    "csi500": ("000905", "中证500"),
}


@dataclass(frozen=True)
class UniverseStock:
    stock_code: str
    stock_name: str = ""
    industry: str = "待补充"
    list_date: str = ""
    universe: str = ""
    source: str = ""


@dataclass
class UniverseResult:
    universe: str
    stocks: list[UniverseStock]
    source: str
    as_of_date: str
    is_historical_membership: bool = False
    warning: str = ""

    @property
    def codes(self) -> list[str]:
        return [item.stock_code for item in self.stocks]


def load_universe(universe: str = DEFAULT_UNIVERSE, limit: int | None = None, timeout: int = 25) -> UniverseResult:
    normalized = normalize_universe_name(universe)
    parts = ["hs300", "csi500"] if normalized == "hs300_csi500" else [normalized]
    stocks: list[UniverseStock] = []
    warnings: list[str] = []
    for part in parts:
        if part == "all_a":
            result = fetch_all_a_current(timeout=timeout)
        elif part in INDEX_SYMBOLS:
            result = fetch_index_current(part, timeout=timeout)
        elif part == "manual":
            result = load_manual_pool()
        else:
            result = load_manual_pool()
            warnings.append(f"未知股票池 {universe}，已使用手工股票池兜底。")
        stocks.extend(result.stocks)
        if result.warning:
            warnings.append(result.warning)

    deduped = dedupe_stocks(stocks)
    if limit:
        deduped = deduped[: max(0, limit)]
    return UniverseResult(
        universe=normalized,
        stocks=deduped,
        source=" + ".join(sorted({item.source for item in deduped if item.source})) or "unknown",
        as_of_date=dt.date.today().isoformat(),
        is_historical_membership=False,
        warning=" ".join(warnings)
        or "当前使用的是接口可获得的最新成分股，不是历史日期成分股。正式历史回测需要接入历史成分股快照，避免成分股未来函数。",
    )


def fetch_index_current(kind: str, timeout: int = 25) -> UniverseResult:
    symbol, name = INDEX_SYMBOLS[kind]
    cached = load_universe_cache(kind)
    fetched = fetch_index_components_with_timeout(symbol, name, timeout=timeout)
    if fetched:
        save_universe_cache(kind, fetched)
        return UniverseResult(kind, fetched, f"{name}当前成分股", current_as_of_date(fetched), False)
    if cached:
        return UniverseResult(
            kind,
            cached,
            f"{name}本地缓存",
            current_as_of_date(cached),
            False,
            f"{name}接口不可用，已使用本地缓存。缓存仍是当前成分股快照，不代表历史成分股。",
        )
    fallback = load_manual_pool().stocks
    return UniverseResult(kind, fallback, "手工股票池兜底", dt.date.today().isoformat(), False, f"{name}接口和缓存均不可用。")


def fetch_all_a_current(timeout: int = 25) -> UniverseResult:
    cached = load_universe_cache("all_a")
    fetched = fetch_all_a_with_timeout(timeout=timeout)
    if fetched:
        save_universe_cache("all_a", fetched)
        return UniverseResult("all_a", fetched, "全A当前列表", current_as_of_date(fetched), False)
    if cached:
        return UniverseResult("all_a", cached, "全A本地缓存", current_as_of_date(cached), False, "全A接口不可用，已使用本地缓存。")
    fallback = load_manual_pool().stocks
    return UniverseResult("all_a", fallback, "手工股票池兜底", dt.date.today().isoformat(), False, "全A接口和缓存均不可用。")


def load_manual_pool() -> UniverseResult:
    stocks = [
        UniverseStock(item.code, item.name, item.sector or "待补充", "", "manual", "stock_pool.csv")
        for item in load_stock_pool_items()
    ]
    return UniverseResult("manual", stocks, "stock_pool.csv", dt.date.today().isoformat(), True)


def fetch_index_components_with_timeout(symbol: str, index_name: str, timeout: int) -> list[UniverseStock]:
    queue: mp.Queue = mp.get_context("spawn").Queue()
    process = mp.get_context("spawn").Process(target=_index_worker, args=(queue, symbol, index_name))
    process.daemon = True
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return []
    if queue.empty():
        return []
    status, payload = queue.get()
    return payload if status == "ok" else []


def _index_worker(queue: mp.Queue, symbol: str, index_name: str) -> None:
    try:
        import akshare as ak

        df = ak.index_stock_cons_csindex(symbol=symbol)
        stocks: list[UniverseStock] = []
        for _, row in df.iterrows():
            code = normalize_code(row.get("成分券代码"))
            if not code:
                continue
            stocks.append(
                UniverseStock(
                    stock_code=code,
                    stock_name=str(row.get("成分券名称") or ""),
                    industry="待补充",
                    list_date=str(row.get("日期") or ""),
                    universe=index_name,
                    source="中证指数当前成分股",
                )
            )
        queue.put(("ok", stocks))
    except Exception as exc:
        queue.put(("error", str(exc)))


def fetch_all_a_with_timeout(timeout: int) -> list[UniverseStock]:
    queue: mp.Queue = mp.get_context("spawn").Queue()
    process = mp.get_context("spawn").Process(target=_all_a_worker, args=(queue,))
    process.daemon = True
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return []
    if queue.empty():
        return []
    status, payload = queue.get()
    return payload if status == "ok" else []


def _all_a_worker(queue: mp.Queue) -> None:
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        stocks: list[UniverseStock] = []
        for _, row in df.iterrows():
            code = normalize_code(row.get("代码"))
            if not code:
                continue
            stocks.append(
                UniverseStock(
                    stock_code=code,
                    stock_name=str(row.get("名称") or ""),
                    industry=str(row.get("行业") or "待补充"),
                    list_date="",
                    universe="全A",
                    source="东方财富当前全A列表",
                )
            )
        queue.put(("ok", stocks))
    except Exception as exc:
        queue.put(("error", str(exc)))


def save_universe_cache(name: str, stocks: list[UniverseStock]) -> None:
    UNIVERSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = universe_cache_path(name)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["stock_code", "stock_name", "industry", "list_date", "universe", "source"])
        writer.writeheader()
        for item in stocks:
            writer.writerow(item.__dict__)


def load_universe_cache(name: str) -> list[UniverseStock]:
    path = universe_cache_path(name)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return [UniverseStock(**row) for row in csv.DictReader(file) if normalize_code(row.get("stock_code"))]
    except Exception:
        return []


def universe_cache_path(name: str) -> Path:
    return UNIVERSE_CACHE_DIR / f"{name}.csv"


def dedupe_stocks(stocks: list[UniverseStock]) -> list[UniverseStock]:
    seen: set[str] = set()
    result: list[UniverseStock] = []
    for item in stocks:
        if item.stock_code in seen:
            continue
        seen.add(item.stock_code)
        result.append(item)
    return result


def current_as_of_date(stocks: list[UniverseStock]) -> str:
    dates = [item.list_date for item in stocks if item.list_date]
    return max(dates) if dates else dt.date.today().isoformat()


def normalize_universe_name(value: str | None) -> str:
    text = (value or DEFAULT_UNIVERSE).strip().lower().replace("-", "_")
    aliases = {
        "沪深300": "hs300",
        "中证500": "csi500",
        "zz500": "csi500",
        "hs300+zz500": "hs300_csi500",
        "hs300+csi500": "hs300_csi500",
        "csi300_csi500": "hs300_csi500",
        "hs300_csi500": "hs300_csi500",
        "all": "all_a",
        "全a": "all_a",
        "全A": "all_a",
    }
    return aliases.get(text, text)


def normalize_code(value: Any) -> str:
    code = str(value or "").strip()
    if code.endswith(".0"):
        code = code[:-2]
    return code if code.isdigit() and len(code) == 6 else ""


if __name__ == "__main__":
    result = load_universe()
    print(f"股票池: {result.universe}, 数量: {len(result.stocks)}, 来源: {result.source}")
    print(result.warning)
