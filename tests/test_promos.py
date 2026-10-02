"""Promo analysis must recover the -7 day distributor shift planted in the demo data."""

import pytest

import analysis as an
from conftest import is_shifted


@pytest.fixture(scope="module")
def weekly_result(sales, promos):
    return an.analyse(sales, promos, an.Settings())


@pytest.fixture(scope="module")
def daily_result(sales, promos):
    return an.analyse(
        sales, promos, an.Settings(freq="D", max_lag=14, baseline_window=35)
    )


def _share_with_lag(metrics, shifted: bool, lags: set) -> float:
    m = metrics[metrics["product"].map(is_shifted) == shifted].dropna(subset=["best_lag"])
    assert len(m) >= 10, "too few series with enough promo events to judge"
    return m["best_lag"].isin(lags).mean()


def test_weekly_portfolio_shift_is_one_week_earlier(weekly_result):
    assert weekly_result.portfolio["shift"] == -1


def test_weekly_per_series_shift(weekly_result):
    m = weekly_result.metrics
    assert _share_with_lag(m, shifted=True, lags={-1}) >= 0.8
    assert _share_with_lag(m, shifted=False, lags={0}) >= 0.8


def test_daily_per_series_shift(daily_result):
    # promo starts are not aligned to weeks, so -8 is an acceptable neighbour of -7
    m = daily_result.metrics
    assert _share_with_lag(m, shifted=True, lags={-7, -8}) >= 0.8
    assert _share_with_lag(m, shifted=False, lags={0}) >= 0.8


def test_shift_adjustment_finds_more_promo_volume(weekly_result):
    p = weekly_result.portfolio
    assert p["promo_volume_share_adj"] > p["promo_volume_share"]
    assert 0 < p["incremental_share"] < p["promo_volume_share_adj"]
    assert p["shift_gain"] > 0


def test_series_without_promos(weekly_result, promos):
    m = weekly_result.metrics.set_index("product")
    unpromoted = set(m.index) - set(promos["product"])
    assert unpromoted, "demo data should contain at least one product without promos"
    assert (m.loc[list(unpromoted), "dependency"] == "no promos").all()


def test_baseline_excludes_promo_periods(weekly_result):
    # promo lifts are large, so a baseline that leaked promo sales would sit far above
    # the median of clean weeks
    panel = weekly_result.panel
    promo_weeks = panel[panel["promo"] & panel["baseline"].notna()]
    assert (promo_weeks["qty"] > promo_weeks["baseline"]).mean() > 0.7
