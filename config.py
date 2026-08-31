from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    pushplus_token: str
    request_timeout: int
    news_timeout: int
    interval_seconds: int
    log_file: Path


def load_settings() -> Settings:
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        pushplus_token=os.getenv("PUSHPLUS_TOKEN", "").strip(),
        request_timeout=int(os.getenv("AI_QUANT_REQUEST_TIMEOUT", "20")),
        news_timeout=int(os.getenv("AI_QUANT_NEWS_TIMEOUT", "20")),
        interval_seconds=int(os.getenv("AI_QUANT_INTERVAL", "10")),
        log_file=Path(os.getenv("AI_QUANT_LOG_FILE", str(BASE_DIR / "logs" / "trading_log.csv"))),
    )
