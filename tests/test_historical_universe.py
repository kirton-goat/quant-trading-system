from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.universe.historical_universe import HistoricalUniverseError, resolve_historical_universe
from research.universe.index_members import IndexMember, IndexSnapshot, save_snapshot, snapshot_path
from research.universe.stock_filter import UniverseFilterConfig, write_rows


def make_snapshot(cache_dir: Path, requested_date: str, snapshot_date: str, codes: list[str]) -> None:
    path = snapshot_path("CSI300", requested_date, cache_dir)
    members = [
        IndexMember(
            requested_date=requested_date,
            snapshot_date=snapshot_date,
            index_code="000300",
            index_name="沪深300",
            code=code,
            name=f"股票{code}",
            source="test historical snapshot",
        )
        for code in codes
    ]
    save_snapshot(IndexSnapshot("CSI300", requested_date, snapshot_date, "test historical snapshot", members, path))


def make_metadata(cache_dir: Path, date: str, rows: list[dict[str, str]]) -> None:
    basics = []
    status = []
    for row in rows:
        basics.append(
            {
                "code": row["code"],
                "name": row.get("name", f"股票{row['code']}"),
                "list_date": row.get("list_date", "2000-01-01"),
                "delist_date": row.get("delist_date", ""),
                "security_type": "1",
                "listing_status": "1",
                "source": "test basics",
            }
        )
        status.append(
            {
                "code": row["code"],
                "status_date": date,
                "name": row.get("name", f"股票{row['code']}"),
                "trade_status": row.get("trade_status", "1"),
                "source": "test status",
            }
        )
    write_rows(cache_dir / "security_basics.csv", basics)
    write_rows(cache_dir / "security_status" / f"{date}.csv", status)


def make_history(date: str, rows: int = 220, amount: float = 50_000_000.0) -> pd.DataFrame:
    dates = pd.bdate_range(end=date, periods=rows)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "stock_code": "000001",
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "volume": 1_000_000.0,
            "amount": amount,
        }
    )


def strict_config() -> UniverseFilterConfig:
    return UniverseFilterConfig(
        min_listing_trading_days=180,
        min_average_amount=20_000_000.0,
        require_market_history=True,
    )


def test_2018_excludes_future_listing(tmp_path: Path) -> None:
    date = "2018-06-29"
    make_snapshot(tmp_path, date, "2018-06-25", ["000001", "300999"])
    make_metadata(
        tmp_path,
        date,
        [
            {"code": "000001", "list_date": "1991-04-03"},
            {"code": "300999", "list_date": "2020-08-24"},
        ],
    )
    histories = {"000001": make_history(date), "300999": make_history(date)}
    result = resolve_historical_universe(
        date,
        "CSI300",
        histories=histories,
        filter_config=strict_config(),
        cache_dir=tmp_path,
        audit_log=tmp_path / "future_data_check.log",
        allow_network=False,
    )
    assert result.codes == ["000001"]
    assert result.filter_stats.future_listing == 1


def test_2020_excludes_already_delisted_stock(tmp_path: Path) -> None:
    date = "2020-06-30"
    make_snapshot(tmp_path, date, "2020-06-29", ["000001", "600999"])
    make_metadata(
        tmp_path,
        date,
        [
            {"code": "000001", "list_date": "1991-04-03"},
            {"code": "600999", "list_date": "2000-01-01", "delist_date": "2019-12-31"},
        ],
    )
    histories = {"000001": make_history(date), "600999": make_history(date)}
    result = resolve_historical_universe(
        date,
        "CSI300",
        histories=histories,
        filter_config=strict_config(),
        cache_dir=tmp_path,
        audit_log=tmp_path / "future_data_check.log",
        allow_network=False,
    )
    assert result.codes == ["000001"]
    assert result.filter_stats.delisted == 1


def test_2025_rejects_future_index_snapshot(tmp_path: Path) -> None:
    date = "2025-06-30"
    make_snapshot(tmp_path, date, "2026-01-01", ["000001"])
    with pytest.raises(HistoricalUniverseError, match="未来"):
        resolve_historical_universe(
            date,
            "CSI300",
            cache_dir=tmp_path,
            audit_log=tmp_path / "future_data_check.log",
            allow_network=False,
        )


def test_missing_2020_snapshot_never_uses_2025_snapshot(tmp_path: Path) -> None:
    make_snapshot(tmp_path, "2025-06-30", "2025-06-30", ["000001"])
    with pytest.raises(HistoricalUniverseError, match="严格模式禁止"):
        resolve_historical_universe(
            "2020-06-30",
            "CSI300",
            cache_dir=tmp_path,
            audit_log=tmp_path / "future_data_check.log",
            allow_network=False,
        )
