import pandas as pd

from research.v3.execution_policy import close_mark_as_of, rebalance_execution_decision


def history(dates):
    return pd.DataFrame({"date": dates, "close": list(range(10, 10 + len(dates)))})


def test_stale_mark_keeps_last_observable_close():
    mark = close_mark_as_of(history(["2020-01-02", "2020-01-03"]), "2020-01-06")
    assert mark.price == 11
    assert mark.observed_on == "2020-01-03"
    assert mark.is_stale is True


def test_rebalance_defers_only_from_observable_execution_close():
    histories = {"000001": history(["2020-01-02"]), "000002": history(["2020-01-02", "2020-01-03"])}
    allowed, locked = rebalance_execution_decision(histories, ["000001", "000002"], "2020-01-03")
    assert allowed is False
    assert locked == ["000001"]
