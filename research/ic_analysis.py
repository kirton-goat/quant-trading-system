from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research.factor_test import FactorObservation


@dataclass
class ICResult:
    factor_name: str
    observations: int
    pearson_ic: float | None
    spearman_ic: float | None
    ic_direction: str
    note: str


def calculate_ic(observations: list[FactorObservation]) -> list[ICResult]:
    factor_names = sorted({item.factor_name for item in observations})
    return [calculate_ic_for_factor(name, observations) for name in factor_names]


def calculate_ic_for_factor(factor_name: str, observations: list[FactorObservation]) -> ICResult:
    rows = [item for item in observations if item.factor_name == factor_name]
    if len(rows) < 5:
        return ICResult(factor_name, len(rows), None, None, "unknown", "样本不足，无法判断IC")

    df = pd.DataFrame([item.__dict__ for item in rows])
    pearson = safe_corr(df, method="pearson")
    spearman = safe_corr(df, method="spearman")
    direction = classify_ic(spearman)
    note = explain_ic(spearman)
    return ICResult(
        factor_name=factor_name,
        observations=len(rows),
        pearson_ic=pearson,
        spearman_ic=spearman,
        ic_direction=direction,
        note=note,
    )


def safe_corr(df: pd.DataFrame, method: str) -> float | None:
    if df["factor_score"].nunique(dropna=True) < 2 or df["forward_return"].nunique(dropna=True) < 2:
        return None
    try:
        if method == "spearman":
            left = df["factor_score"].rank(method="average")
            right = df["forward_return"].rank(method="average")
            value = left.corr(right, method="pearson")
        else:
            value = df["factor_score"].corr(df["forward_return"], method=method)
    except Exception:
        return None
    if pd.isna(value):
        return None
    return round(float(value), 4)


def classify_ic(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 0.05:
        return "positive"
    if value <= -0.05:
        return "negative"
    return "weak"


def explain_ic(value: float | None) -> str:
    if value is None:
        return "IC不可计算"
    if value >= 0.08:
        return "因子与未来收益正相关较明显"
    if value >= 0.03:
        return "因子与未来收益存在弱正相关"
    if value <= -0.08:
        return "因子可能反向有效或当前评分方向有误"
    if value <= -0.03:
        return "因子与未来收益存在弱负相关"
    return "IC接近0，暂未体现稳定预测能力"
