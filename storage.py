from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


HEADERS = [
    "时间",
    "新闻来源",
    "事件类型",
    "新闻标题",
    "关联股票",
    "当前价格",
    "涨跌幅",
    "行情来源",
    "技术趋势",
    "技术摘要",
    "20日涨幅",
    "60日涨幅",
    "5/20量能比",
    "跳空幅度",
    "换手率",
    "information_type",
    "source_level",
    "source_score",
    "is_original",
    "is_repeat",
    "has_financial_impact",
    "market_already_priced",
    "trade_value",
    "新闻质量评分",
    "新闻交易价值评分",
    "技术指标评分",
    "AI评分",
    "AI原始操作",
    "质量阈值拦截",
    "多因子综合评分",
    "动量因子",
    "资金因子",
    "事件因子",
    "技术因子",
    "基本面因子",
    "市场环境因子",
    "多因子交易许可",
    "多因子拦截原因",
    "综合评分",
    "AI操作",
    "AI逻辑",
    "风险标签",
    "signal_id",
    "sim_status",
    "sim_direction",
    "sim_entry_time",
    "sim_entry_price",
    "sim_exit_time",
    "sim_exit_price",
    "sim_holding_days",
    "sim_pnl_pct",
    "sim_pnl_amount",
    "sim_result",
    "sim_note",
]


def append_log(log_file: Path, row: dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ensure_compatible_header(log_file)
    exists = log_file.exists()
    with log_file.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"记录已保存: {log_file}")


def ensure_compatible_header(log_file: Path) -> None:
    if not log_file.exists() or log_file.stat().st_size == 0:
        return

    with log_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        current_header = next(reader, [])

    if current_header == HEADERS:
        return

    if current_header and all(header in HEADERS for header in current_header):
        with log_file.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        with log_file.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=HEADERS)
            writer.writeheader()
            for row in rows:
                writer.writerow({header: row.get(header, "") for header in HEADERS})
        print(f"日志表头已兼容升级: {log_file}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = log_file.with_name(f"{log_file.stem}_legacy_{timestamp}{log_file.suffix}")
    log_file.replace(backup)
    print(f"检测到日志表头变化，旧日志已备份: {backup}")
