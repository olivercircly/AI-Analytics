import pandas as pd
import pytest

import calendar_effects as ce
import seasonality as se


@pytest.fixture(scope="module")
def holiday_summary(daily):
    effects = ce.holiday_effects(daily.groupby("period")["qty"].sum(), "AT")
    return ce.summarise(effects)


def test_bridge_days():
    cal = ce.holiday_calendar("AT", [2025])
    bridges = cal.loc[cal["name"] == "Bridge day", "date"]
    # Ascension (Thu 29 May) and Corpus Christi (Thu 19 June) 2025
    assert {pd.Timestamp("2025-05-30"), pd.Timestamp("2025-06-20")} <= set(bridges)


@pytest.mark.parametrize("country", sorted(ce.COUNTRIES))
def test_supported_calendars_load(country):
    cal = ce.holiday_calendar(country, [2025])
    assert cal["public"].any()


def test_demo_shops_closed_on_holidays(holiday_summary):
    assert {"Christmas Day", "New Year's Day", "Easter Monday"} <= set(holiday_summary["closed"])


def test_demo_busier_day_before_holiday(holiday_summary):
    _, offset, effect = holiday_summary["biggest_uplift"]
    assert offset == -1
    assert effect > 0.2


def test_seasonality_classify_thresholds():
    got = se.classify(pd.Series([0.1, 0.45, 0.9, float("nan")])).tolist()
    assert got == ["weak", "moderate", "strong", "too short"]


def test_demo_weekday_seasonality_is_strong(daily, weekly):
    p = se.analyse(daily, weekly)["portfolio"]
    assert p["weekday_strength"] > 0.6
    assert p["seasonal_yearly_volume"] > 0.5
