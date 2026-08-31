from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EVENT_DATA_DIR = BASE_DIR / "data_cache" / "events"
POLICY_EVENT_FILE = EVENT_DATA_DIR / "policy_events.csv"
ANNOUNCEMENT_EVENT_FILE = EVENT_DATA_DIR / "announcement_events.csv"
EVENT_COLUMNS = [
    "published_at", "event_type", "stock_code", "industry", "score", "source", "publisher",
    "source_url", "source_kind", "is_official", "fetched_at", "content_hash", "title", "note"
]


@dataclass(frozen=True)
class EventResearchStats:
    policy_rows: int = 0
    announcement_rows: int = 0

    @property
    def total_rows(self) -> int:
        return self.policy_rows + self.announcement_rows


def ensure_event_templates() -> None:
    EVENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (POLICY_EVENT_FILE, ANNOUNCEMENT_EVENT_FILE):
        if not path.exists():
            pd.DataFrame(columns=EVENT_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def record_research_event(
    event_type: str,
    published_at: str,
    stock_code: str | None,
    score: float,
    source: str,
    title: str,
    content: str = "",
    publisher: str = "",
    source_url: str = "",
    source_kind: str = "aggregated_news",
    is_official: bool = False,
) -> bool:
    """Persist only Level 4/5 policy or announcement observations for later tests."""
    if event_type.startswith("政策信息/"):
        path, kind = POLICY_EVENT_FILE, "policy"
    elif event_type.startswith("公司公告/"):
        path, kind = ANNOUNCEMENT_EVENT_FILE, "announcement"
    else:
        return False
    ensure_event_templates()
    timestamp = pd.to_datetime(published_at, errors="coerce")
    if pd.isna(timestamp):
        return False
    existing = _load(path)
    code = "" if stock_code is None else str(stock_code)
    content_hash = _content_hash(source_url or f"{published_at}|{source}|{title}")
    duplicate = (
        (existing["published_at"].astype(str) == timestamp.strftime("%Y-%m-%d"))
        & (existing["stock_code"].astype(str) == code)
        & (existing["title"].astype(str) == str(title))
    )
    if duplicate.any() or existing["content_hash"].astype(str).eq(content_hash).any():
        return False
    row = pd.DataFrame(
        [{
            "published_at": timestamp.strftime("%Y-%m-%d"),
            "event_type": kind,
            "stock_code": code,
            "industry": extract_industry(content),
            "score": max(0.0, min(float(score), 100.0)),
            "source": source,
            "publisher": publisher or source,
            "source_url": source_url,
            "source_kind": source_kind,
            "is_official": bool(is_official),
            "fetched_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "content_hash": content_hash,
            "title": title,
            "note": "由事件扫描自动记录；仅供研究与后续增量验证，不进入当前交易评分。",
        }]
    )
    pd.concat([existing, row], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    return True


def load_event_data() -> tuple[pd.DataFrame, EventResearchStats]:
    ensure_event_templates()
    policy = _load(POLICY_EVENT_FILE)
    announcements = _load(ANNOUNCEMENT_EVENT_FILE)
    combined = pd.concat([policy, announcements], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS), EventResearchStats()
    combined["published_at"] = pd.to_datetime(combined["published_at"], errors="coerce")
    combined["stock_code"] = combined["stock_code"].fillna("").astype(str).str.extract(r"(\d{6})", expand=False).fillna("")
    combined["industry"] = combined["industry"].fillna("").astype(str)
    combined["event_type"] = combined["event_type"].fillna("").astype(str)
    combined["score"] = pd.to_numeric(combined["score"], errors="coerce").clip(0, 100).fillna(50.0)
    return combined.dropna(subset=["published_at"]).sort_values("published_at"), EventResearchStats(len(policy), len(announcements))


def event_scores_as_of(
    as_of_date: str,
    stock_codes: list[str],
    code_industries: dict[str, str] | None = None,
    lookback_days: int = 20,
) -> tuple[dict[str, float], EventResearchStats]:
    events, stats = load_event_data()
    scores = {code: 50.0 for code in stock_codes}
    if events.empty:
        return scores, stats
    as_of = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(as_of):
        return scores, stats
    recent = events[(events["published_at"] <= as_of) & (events["published_at"] >= as_of - pd.Timedelta(days=lookback_days))]
    for code in stock_codes:
        industry = (code_industries or {}).get(code, "")
        matched = recent[(recent["stock_code"] == code)]
        if matched.empty and industry:
            matched = recent[(recent["stock_code"] == "") & (recent["industry"] == industry)]
        if matched.empty:
            continue
        newest = matched.sort_values("published_at").iloc[-1]
        age = max(0, (as_of.normalize() - newest["published_at"].normalize()).days)
        decay = max(0.2, 1 - age / max(lookback_days, 1))
        scores[code] = round(50 + (float(newest["score"]) - 50) * decay, 2)
    return scores, stats


def _load(path: Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    for column in EVENT_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    return data[EVENT_COLUMNS]


def _content_hash(value: str) -> str:
    import hashlib
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()[:24]


def extract_industry(content: str) -> str:
    marker = "影响行业："
    if marker not in content:
        return ""
    value = content.split(marker, 1)[1].split("；", 1)[0].split("\n", 1)[0].strip()
    return value.split(",", 1)[0].strip()
