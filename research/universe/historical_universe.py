from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .index_members import (
    DEFAULT_CACHE_DIR,
    HistoricalIndexDataError,
    IndexSnapshot,
    get_index_snapshot,
    normalize_date,
)
from .stock_filter import (
    UniverseFilterConfig,
    UniverseFilterStats,
    filter_universe,
    load_security_metadata,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_LOG = BASE_DIR / "future_data_check.log"


class HistoricalUniverseError(RuntimeError):
    pass


@dataclass
class HistoricalUniverseResult:
    date: str
    universe_type: str
    stocks: list[dict[str, Any]]
    source: str
    snapshot_dates: list[str]
    raw_count: int
    filter_stats: UniverseFilterStats = field(default_factory=UniverseFilterStats)

    @property
    def codes(self) -> list[str]:
        return [item["code"] for item in self.stocks]


def get_historical_universe(
    date: str | dt.date | dt.datetime,
    universe_type: str = "CSI300",
    histories: dict[str, pd.DataFrame] | None = None,
    filter_config: UniverseFilterConfig | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    audit_log: Path = DEFAULT_AUDIT_LOG,
    allow_network: bool = True,
) -> list[dict[str, Any]]:
    return resolve_historical_universe(
        date=date,
        universe_type=universe_type,
        histories=histories,
        filter_config=filter_config,
        cache_dir=cache_dir,
        audit_log=audit_log,
        allow_network=allow_network,
    ).stocks


def resolve_historical_universe(
    date: str | dt.date | dt.datetime,
    universe_type: str = "CSI300",
    histories: dict[str, pd.DataFrame] | None = None,
    filter_config: UniverseFilterConfig | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    audit_log: Path = DEFAULT_AUDIT_LOG,
    allow_network: bool = True,
) -> HistoricalUniverseResult:
    requested_date = normalize_date(date)
    parts = normalize_universe_parts(universe_type)
    snapshots: list[IndexSnapshot] = []
    try:
        for part in parts:
            snapshots.append(get_index_snapshot(requested_date, part, cache_dir, allow_network=allow_network))
    except HistoricalIndexDataError as exc:
        append_audit(audit_log, requested_date, universe_type, "BLOCK", str(exc))
        raise HistoricalUniverseError(str(exc)) from exc

    members = dedupe_members(
        [
            {
                "code": member.code,
                "name": member.name,
                "sector": member.sector,
                "source": member.source,
                "status": member.status,
                "universe": member.index_name,
                "index_code": member.index_code,
                "snapshot_date": member.snapshot_date,
            }
            for snapshot in snapshots
            for member in snapshot.members
        ]
    )
    metadata = load_security_metadata(requested_date, cache_dir, allow_network=allow_network)
    config = filter_config or UniverseFilterConfig(require_market_history=histories is not None)
    filtered, stats = filter_universe(members, requested_date, histories, metadata, config)

    violations = stats.future_membership + stats.future_listing
    if violations:
        detail = f"检测到 {violations} 条未来数据，已全部阻断。"
        append_audit(audit_log, requested_date, universe_type, "BLOCK", detail)
    else:
        append_audit(
            audit_log,
            requested_date,
            universe_type,
            "PASS",
            f"raw={len(members)} filtered={len(filtered)} snapshots={','.join(snapshot.snapshot_date for snapshot in snapshots)}",
        )
    return HistoricalUniverseResult(
        date=requested_date,
        universe_type="+".join(parts),
        stocks=filtered,
        source=" + ".join(snapshot.source for snapshot in snapshots),
        snapshot_dates=sorted({snapshot.snapshot_date for snapshot in snapshots}),
        raw_count=len(members),
        filter_stats=stats,
    )


def normalize_universe_parts(value: str) -> list[str]:
    normalized = str(value or "CSI300").strip().upper().replace("-", "_").replace("+", "_")
    if normalized in {"CSI300_CSI500", "HS300_CSI500", "HS300_ZZ500", "CSI800", "000906"}:
        return ["CSI300", "CSI500"]
    if normalized in {"CSI300", "HS300", "000300", "沪深300"}:
        return ["CSI300"]
    if normalized in {"CSI500", "ZZ500", "000905", "中证500"}:
        return ["CSI500"]
    raise HistoricalUniverseError(f"暂不支持历史股票池类型：{value}")


def dedupe_members(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for member in members:
        code = str(member.get("code") or "")
        if not code:
            continue
        if code in result:
            universes = set(str(result[code].get("universe") or "").split("+"))
            universes.add(str(member.get("universe") or ""))
            result[code]["universe"] = "+".join(sorted(item for item in universes if item))
        else:
            result[code] = dict(member)
    return list(result.values())


def append_audit(path: Path, date: str, universe_type: str, result: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{timestamp}\tbacktest_date={date}\tuniverse={universe_type}\tresult={result}\t{detail}\n")
