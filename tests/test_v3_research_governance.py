from pathlib import Path

import pytest

from research.experiments.registry import ExperimentRegistryError, ExperimentRequest, create_experiment, list_experiments
from research.v3.preflight import default_v3_config, required_history_start
from research.v3.runner import V3DataIntegrityError, queue_v3_experiment
from research.v3.data_acquisition import normalize_qfq_history
from research.v3.fundamental_acquisition import valid_fundamental_rows
from research.v3.corporate_action_validation import qfq_return_on_or_after


def request() -> ExperimentRequest:
    return ExperimentRequest(
        hypothesis_note="Test whether momentum has independent cross-sectional value.",
        strategy_version="v2_continuous_rebalance",
        data_version="research_cache_v3",
        sample_period={"start": "2015-01-01", "end": "2025-12-31"},
        factor_weights={"momentum": 0.5, "fundamental": 0.5},
        enabled_factors=["momentum", "fundamental"],
        market_regime_gate=False,
        execution_assumptions={"execution": "next_trading_day_close"},
        fee_assumptions={"fee_rate": 0.0015},
        benchmark_setup={"CSI300": "sh000300", "CSI500": "sh000905"},
    )


def test_registry_is_append_only_and_preserves_multiple_attempts(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"
    first = create_experiment(request(), path)
    second = create_experiment(request(), path)
    records = list_experiments(path)
    assert first.experiment_id != second.experiment_id
    assert len(records) == 2
    assert records[0].configuration_hash == records[1].configuration_hash


def test_registry_rejects_missing_hypothesis(tmp_path: Path) -> None:
    bad = request()
    bad = ExperimentRequest(**{**bad.__dict__, "hypothesis_note": ""})
    with pytest.raises(ExperimentRegistryError):
        create_experiment(bad, tmp_path / "experiments.jsonl")


def test_v3_configuration_keeps_formal_weights_and_requires_long_lookback() -> None:
    config = default_v3_config()
    assert config["result_classification"] == "research_experiment"
    assert sum(config["factor_weights"].values()) == 1.0
    assert required_history_start("2015-01-01") < "2013-01-01"


def test_v3_runner_blocks_unvalidated_long_sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from research.v3 import runner
    from research.v3.preflight import V3PreflightResult

    monkeypatch.setattr(runner, "create_experiment", lambda request: type("Record", (), {"experiment_id": "exp_test"})())
    monkeypatch.setattr(runner, "update_experiment_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "write_preflight", lambda result: None)
    monkeypatch.setattr(
        runner, "run_preflight",
        lambda config: V3PreflightResult("2015-01-01", "2025-12-31", "2012-07-15", 1, False, False, False, False, True, "incomplete"),
    )
    with pytest.raises(V3DataIntegrityError):
        queue_v3_experiment("Do not run invalid long sample.")


def test_v3_strategy_price_cache_rows_record_adjustment_and_source() -> None:
    raw = __import__("pandas").DataFrame({
        "日期": ["2020-01-02"], "开盘": [10], "最高": [11], "最低": [9], "收盘": [10.5], "成交量": [100], "成交额": [1000],
    })
    output = normalize_qfq_history(raw, "000001", "test-source")
    assert output.iloc[0]["adjustment"] == "qfq"
    assert output.iloc[0]["source"] == "test-source"


def test_v3_fundamental_preparation_quarantines_invalid_disclosure_dates() -> None:
    frame = __import__("pandas").DataFrame({
        "code": ["000001", "000001"], "report_period": ["2005-12-31", "2015-12-31"],
        "disclosure_date": ["1900-01-01", "2016-03-30"],
    })
    valid, invalid = valid_fundamental_rows(frame)
    assert len(valid) == 1
    assert len(invalid) == 1


def test_corporate_action_check_uses_first_trading_day_on_or_after_event() -> None:
    price = __import__("pandas").DataFrame({"date": ["2020-01-01", "2020-01-02", "2020-01-03"], "close": [10, 10.2, 10.3]})
    date, value = qfq_return_on_or_after(price, __import__("pandas").Timestamp("2020-01-02"))
    assert date == "2020-01-02"
    assert round(value, 4) == 0.02
