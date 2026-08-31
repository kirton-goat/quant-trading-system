from __future__ import annotations

import csv
import datetime as dt
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = BASE_DIR / "data_cache" / "historical_universe"

INDEX_DEFINITIONS = {
    "CSI300": ("000300", "沪深300", "query_hs300_stocks"),
    "CSI500": ("000905", "中证500", "query_zz500_stocks"),
}

ALIASES = {
    "HS300": "CSI300",
    "000300": "CSI300",
    "沪深300": "CSI300",
    "CSI300": "CSI300",
    "ZZ500": "CSI500",
    "000905": "CSI500",
    "中证500": "CSI500",
    "CSI500": "CSI500",
}

SNAPSHOT_COLUMNS = [
    "requested_date",
    "snapshot_date",
    "index_code",
    "index_name",
    "code",
    "name",
    "sector",
    "source",
    "status",
]


class HistoricalIndexDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexMember:
    requested_date: str
    snapshot_date: str
    index_code: str
    index_name: str
    code: str
    name: str
    sector: str = "待补充"
    source: str = ""
    status: str = "active"


@dataclass(frozen=True)
class IndexSnapshot:
    universe_type: str
    requested_date: str
    snapshot_date: str
    source: str
    members: list[IndexMember]
    cache_path: Path


def get_index_snapshot(
    date: str | dt.date | dt.datetime,
    universe_type: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    allow_network: bool = True,
) -> IndexSnapshot:
    requested_date = normalize_date(date)
    normalized_type = normalize_index_type(universe_type)
    path = snapshot_path(normalized_type, requested_date, cache_dir)
    cached = load_snapshot(path, normalized_type, requested_date)
    if cached is not None:
        return cached
    if not allow_network:
        raise HistoricalIndexDataError(
            f"缺少 {normalized_type} 在 {requested_date} 的历史成分快照；严格模式禁止使用当前成分股兜底。"
        )
    snapshot = fetch_baostock_snapshot(requested_date, normalized_type, path)
    save_snapshot(snapshot)
    return snapshot


def fetch_baostock_snapshot(requested_date: str, universe_type: str, path: Path) -> IndexSnapshot:
    try:
        import baostock as bs
    except ImportError as exc:
        raise HistoricalIndexDataError("缺少 baostock 依赖，无法按日期获取历史指数成分。") from exc

    index_code, index_name, method_name = INDEX_DEFINITIONS[universe_type]
    login = bs.login()
    if login.error_code != "0":
        raise HistoricalIndexDataError(f"BaoStock 登录失败：{login.error_msg}")
    try:
        query = getattr(bs, method_name)
        result = query(date=requested_date)
        if result.error_code != "0":
            raise HistoricalIndexDataError(
                f"BaoStock 获取 {index_name} {requested_date} 成分失败：{result.error_msg}"
            )
        rows: list[dict[str, Any]] = []
        while result.next():
            rows.append(dict(zip(result.fields, result.get_row_data())))
    finally:
        bs.logout()

    source = f"BaoStock {method_name}(date={requested_date})"
    members: list[IndexMember] = []
    for row in rows:
        code = normalize_code(row.get("code"))
        if not code:
            continue
        snapshot_date = normalize_date(row.get("updateDate") or requested_date)
        if snapshot_date > requested_date:
            raise HistoricalIndexDataError(
                f"数据源返回未来快照：请求 {requested_date}，返回 {snapshot_date}，已拒绝写入。"
            )
        members.append(
            IndexMember(
                requested_date=requested_date,
                snapshot_date=snapshot_date,
                index_code=index_code,
                index_name=index_name,
                code=code,
                name=str(row.get("code_name") or "").strip(),
                source=source,
            )
        )
    expected_count = 300 if universe_type == "CSI300" else 500
    if len(members) < expected_count * 0.9:
        raise HistoricalIndexDataError(
            f"{index_name} {requested_date} 仅返回 {len(members)} 只，低于完整快照最低要求，已拒绝使用。"
        )
    snapshot_date = max(item.snapshot_date for item in members)
    return IndexSnapshot(universe_type, requested_date, snapshot_date, source, members, path)


def save_snapshot(snapshot: IndexSnapshot) -> None:
    snapshot.cache_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.cache_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SNAPSHOT_COLUMNS)
        writer.writeheader()
        for member in snapshot.members:
            writer.writerow(asdict(member))


def load_snapshot(path: Path, universe_type: str, requested_date: str) -> IndexSnapshot | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            members = [IndexMember(**row) for row in csv.DictReader(file)]
    except (OSError, TypeError, ValueError) as exc:
        raise HistoricalIndexDataError(f"历史成分快照损坏：{path}: {exc}") from exc
    if not members:
        raise HistoricalIndexDataError(f"历史成分快照为空：{path}")
    if any(item.requested_date != requested_date for item in members):
        raise HistoricalIndexDataError(f"快照请求日期与文件名不一致：{path}")
    future_dates = [item.snapshot_date for item in members if item.snapshot_date > requested_date]
    if future_dates:
        raise HistoricalIndexDataError(f"快照包含未来生效日期 {min(future_dates)}：{path}")
    snapshot_date = max(item.snapshot_date for item in members)
    source = members[0].source
    return IndexSnapshot(universe_type, requested_date, snapshot_date, source, members, path)


def snapshot_path(universe_type: str, requested_date: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / "index_members" / universe_type.lower() / f"{requested_date}.csv"


def normalize_index_type(value: str) -> str:
    normalized = str(value or "").strip().upper().replace("-", "").replace("_", "")
    result = ALIASES.get(normalized)
    if result is None:
        raise HistoricalIndexDataError(f"暂不支持历史股票池类型：{value}")
    return result


def normalize_date(value: str | dt.date | dt.datetime | Any) -> str:
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise HistoricalIndexDataError(f"无效日期：{value}")


def normalize_code(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "." in text:
        text = text.split(".")[-1]
    if text.endswith(".0"):
        text = text[:-2]
    return text if text.isdigit() and len(text) == 6 else ""
