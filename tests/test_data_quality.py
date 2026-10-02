"""Data quality must find the problems planted in the demo data: one system-wide outage,
lost and half-loaded days per account, a data-entry spike and a silent account."""

import pandas as pd
import pytest

import data_quality as dq

ACCOUNTS = {"1404", "1405", "1406"}


@pytest.fixture(scope="module")
def result(sales, daily, quality, holiday_dates, today):
    return dq.analyse(sales, daily, quality, holiday_dates, today)


def test_one_system_wide_outage(result):
    outages = result["outages"]
    assert len(outages) == 1
    assert outages["missing"].iloc[0] == len(ACCOUNTS)


def test_lost_days_flagged_missing_per_account(result):
    # outage + 2 lost days per account (a lost day can coincide with the outage)
    missing = result["flagged"].query("flag == 'missing'").groupby("location").size()
    assert set(missing.index) == ACCOUNTS
    assert (missing >= 2).all() and (missing <= 3).all()


def test_half_loaded_days_flagged_partial(result):
    partial = result["flagged"].query("flag == 'partial'").groupby("location").size()
    assert partial.to_dict() == {a: 2 for a in ACCOUNTS}


def test_holidays_and_sundays_not_flagged(result, holiday_dates):
    flagged = result["flagged"]
    assert not flagged["date"].isin(holiday_dates).any()
    assert not (flagged["date"].dt.dayofweek == 6).any()


def test_silent_account(result):
    silent = result["summary"]["silent_accounts"]
    assert silent["location"].tolist() == ["1406"]
    assert silent["days_silent"].iloc[0] >= 20


def test_data_entry_spike_is_top_outlier(result, sales):
    top = result["outliers"].iloc[0]
    assert top["product"] == "009705"
    assert top["period"] == sales["date"].min() + pd.Timedelta(days=500)
    assert top["kind"] == "spike"


def test_latency(result):
    lat = result["latency"]
    # make_demo_quality: 35% of rows arrive within a day, 75% within two, 85% within three
    assert lat["k50"] == 1
    assert lat["k80"] == 2
    assert not lat["bulk_loaded"]


def test_backfill_before_onboarding(quality):
    a = dq.arrivals(quality)
    backfill = a[a["kind"] == "backfill / re-import"]
    assert backfill["arrival"].nunique() == 1
    assert backfill["arrival"].iloc[0] == pd.Timestamp("2025-01-06")
