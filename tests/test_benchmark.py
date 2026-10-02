import numpy as np
import pandas as pd
import pytest

import benchmark as bm


@pytest.fixture(scope="module")
def result(weekly):
    return bm.backtest(weekly, "W-MON", horizon=1, test_periods=26)


def test_forecasts_use_only_past_data():
    periods = pd.date_range("2024-01-01", periods=30, freq="D")
    panel = pd.DataFrame({"product": "A", "period": periods, "qty": np.arange(30.0)})
    fc = bm.forecasts(panel, "D", horizon=2)
    assert fc["Naive"].iloc[10] == 8
    assert fc["Same weekday last week"].iloc[10] == 3
    # the moving average must not include the 2 periods it is forecasting over
    assert fc["Moving average (28 d)"].iloc[29] == np.arange(0, 28).mean()


def test_methods_compared_on_same_rows(result):
    by_method = result["by_method"]
    assert len(by_method) == 4
    assert by_method["actual"].nunique() == 1


def test_portfolio_metrics(result):
    p = result["portfolio"]
    assert p["best_method"] == result["by_method"]["method"].iloc[0]
    assert 0 < p["wape"] < 1
    assert p["total_wape"] < p["wape"]  # errors cancel when aggregated
    assert p["articles_tested"] == 40
