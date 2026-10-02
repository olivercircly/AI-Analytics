"""Article anomalies must find what demo.make_demo_data plants in single articles,
and stay quiet on the clean ones."""

import pandas as pd
import pytest

import analysis as an
import anomalies as ab

WEEKLY = an.Settings()
DAILY = an.Settings(freq="D", max_lag=14, baseline_window=35)


@pytest.fixture(scope="module")
def article(sales, promos, holiday_dates):
    end = sales["date"].max()

    def run(product: str, s: an.Settings = WEEKLY) -> dict:
        # like the DB: one article, no rows for zero sales
        sub = sales[(sales["product"] == product) & (sales["qty"] != 0)]
        return ab.analyse(sub, promos[promos["product"] == product], s, holiday_dates, end)

    return run


def _day(sales, n: int) -> pd.Timestamp:
    return sales["date"].min() + pd.Timedelta(days=n)


def _kinds(result: dict) -> list[str]:
    return result["events"]["kind"].tolist()


@pytest.mark.parametrize("s", [WEEKLY, DAILY], ids=["weekly", "daily"])
def test_spike(article, sales, s):
    ev = article("009705", s)["events"]
    spikes = ev[ev["kind"] == "spike"]
    assert len(spikes) == 1
    assert spikes["start"].iloc[0] == an.to_period(pd.Series([_day(sales, 500)]), s.freq)[0]


@pytest.mark.parametrize("s", [WEEKLY, DAILY], ids=["weekly", "daily"])
def test_level_shift(article, sales, s):
    r = article("009707", s)
    ls = r["level_shifts"]
    assert len(ls) == 1
    assert abs((ls["start"].iloc[0] - _day(sales, 700)).days) <= 7
    assert ls["change"].iloc[0] < -0.3
    assert ls["account"].iloc[0] == "1405"


def test_stock_out_daily(article, sales):
    gaps = article("009711", DAILY)["events"].query("kind == 'gap'")
    assert len(gaps) == 1
    g = gaps.iloc[0]
    # planted: days 400-420; the next day may happen to be zero as well
    assert g["start"] == _day(sales, 400)
    assert _day(sales, 420) <= g["end"] <= _day(sales, 422)


def test_stock_out_weekly_doesnt_become_level_shifts(article):
    r = article("009711")
    assert "gap" in _kinds(r)
    assert r["level_shifts"].empty


def test_stopped_reported_once(article, sales):
    ev = article("009702")["events"]  # account 1406 stops 20 days before the end
    stopped = ev[ev["kind"] == "stopped"]
    assert len(stopped) == 1
    assert stopped["account"].iloc[0] == "all"
    assert stopped["start"].iloc[0] >= sales["date"].max() - pd.Timedelta(days=21)


@pytest.mark.parametrize("s", [WEEKLY, DAILY], ids=["weekly", "daily"])
def test_clean_articles_stay_quiet(article, sales, s):
    # articles at accounts 1404/1405 without planted article anomalies; daily gaps are
    # excluded because the planted outage and lost days are real gaps for every article
    clean = [p for i, p in enumerate(sorted(sales["product"].unique())) if i % 3 != 2]
    clean = [p for p in clean if p not in {"009705", "009707"}]
    noise = {"spike", "drop", "level shift", "stopped"}
    flagged = [p for p in clean if noise & set(_kinds(article(p, s)))]
    assert len(flagged) <= 0.1 * len(clean), flagged


def test_flat_promo_needs_enough_expected_sales():
    days = pd.date_range("2024-01-01", periods=120, freq="D")
    panel = pd.DataFrame({"period": days, "qty": 1.0, "promo": False})
    expected = pd.Series(1.0, index=panel.index)
    promos = pd.DataFrame(
        {
            "product": "A",
            "location": [None, None],
            "promo_start": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "promo_end": pd.to_datetime(["2024-02-07", "2024-03-30"]),
        }
    )
    # both show no uplift, but only the 30-day promo expected enough units to judge
    checks = ab.promo_check(panel, expected, promos, "D", shift=0)
    assert checks["flat"].tolist() == [False, True]


def test_promo_lift_is_per_promo_day():
    # weekly: a 2-day promo doubling sales on those days must show lift 2, not 1.3
    weeks = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    panel = pd.DataFrame({"period": weeks, "qty": 70.0, "promo": False})
    panel.loc[4, "qty"] = 70 + 20
    promos = pd.DataFrame(
        {
            "product": ["A"],
            "location": [None],
            "promo_start": [weeks[4] + pd.Timedelta(days=2)],
            "promo_end": [weeks[4] + pd.Timedelta(days=3)],
        }
    )
    checks = ab.promo_check(panel, pd.Series(70.0, index=panel.index), promos, "W-MON", 0)
    assert checks["lift"].iloc[0] == pytest.approx(2.0)


def test_expected_level_survives_long_zero_run():
    # slow mover (1/day) with a 3-week stock-out: the median would be 0 and hide it
    days = pd.date_range("2024-01-01", periods=140, freq="D")
    qty = pd.Series([1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0] * 20)
    qty[60:81] = 0
    panel = pd.DataFrame({"period": days, "qty": qty, "promo": False})
    exp = ab.expected_level(panel, "D", 35, panel["promo"])
    assert exp[60:81].sum() > 10
    assert len(ab.gaps(panel, exp)) == 1
