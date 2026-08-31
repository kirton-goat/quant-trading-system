from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FUNDAMENTAL_CACHE = BASE_DIR / "data_cache" / "fundamentals"


def cache_path(code: str, cache_dir: Path = DEFAULT_FUNDAMENTAL_CACHE) -> Path:
    normalized = "".join(character for character in str(code) if character.isdigit()).zfill(6)
    return cache_dir / f"{normalized}.csv"


def read_fundamental_cache(code: str, cache_dir: Path = DEFAULT_FUNDAMENTAL_CACHE) -> pd.DataFrame:
    path = cache_path(code, cache_dir)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def write_fundamental_cache(
    code: str,
    data: pd.DataFrame,
    cache_dir: Path = DEFAULT_FUNDAMENTAL_CACHE,
) -> Path:
    path = cache_path(code, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = data.copy()
    output["fetch_time"] = dt.datetime.now(dt.timezone.utc).isoformat()
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return path
