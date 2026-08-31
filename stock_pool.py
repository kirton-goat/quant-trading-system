from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STOCK_POOL_FILE = BASE_DIR / "stock_pool.csv"


@dataclass(frozen=True)
class StockPoolItem:
    code: str
    name: str = ""
    sector: str = ""
    weight: float = 1.0
    enabled: bool = True
    note: str = ""


def load_stock_pool(pool_file: Path | str = DEFAULT_STOCK_POOL_FILE, limit: int | None = None) -> list[str]:
    return [item.code for item in load_stock_pool_items(pool_file, limit=limit) if item.enabled]


def load_stock_pool_items(pool_file: Path | str = DEFAULT_STOCK_POOL_FILE, limit: int | None = None) -> list[StockPoolItem]:
    path = Path(pool_file)
    if not path.exists():
        return []

    items: list[StockPoolItem] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                item = build_item(row)
                if not item or not item.enabled:
                    continue
                if item.code not in seen:
                    items.append(item)
                    seen.add(item.code)
                if limit and len(items) >= limit:
                    break
    except Exception:
        return []
    return items


def build_item(row: dict[str, str]) -> StockPoolItem | None:
    code = normalize_code(row.get("code") or row.get("股票代码") or "")
    if not code:
        return None
    return StockPoolItem(
        code=code,
        name=(row.get("name") or row.get("股票名称") or "").strip(),
        sector=(row.get("sector") or row.get("行业") or "").strip(),
        weight=parse_float(row.get("weight"), default=1.0),
        enabled=parse_enabled(row.get("enabled")),
        note=(row.get("note") or row.get("备注") or "").strip(),
    )


def normalize_code(value: str) -> str:
    code = str(value).strip()
    if code.endswith(".0"):
        code = code[:-2]
    return code if code.isdigit() and len(code) == 6 else ""


def parse_enabled(value: str | None) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text not in {"0", "false", "no", "n", "否", "停用"}


def parse_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except Exception:
        return default


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    codes = load_stock_pool()
    print(f"股票池数量: {len(codes)}")
    print(",".join(codes))
