"""Data loading and cached analysis results.

src = (cid, start, promo_window, demo) identifies the data; everything is cached on it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import analysis as an
import benchmark as bm
import calendar_effects as ce
import data_quality as dq
import seasonality as se


@st.cache_resource(ttl=3600, show_spinner="Loading data…")
def load(src: tuple) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cid, start, promo_window, demo = src
    if demo:
        from demo import make_demo_data

        sales, promos = make_demo_data()
        return (
            sales[sales["date"] >= pd.Timestamp(start)],
            promos,
            {"promo_rows": len(promos), "secondary_rows": 0},
        )
    import queries as q  # imported lazily so demo mode works without DB credentials

    raw = q.load_promos(cid, start)
    info = {
        "promo_rows": len(raw),
        "secondary_rows": int(raw["secondary_start"].notna().sum()),
    }
    sales, promos = q.label_articles(
        q.load_sales(cid, start), q.select_window(raw, promo_window)
    )
    return sales, promos, info


@st.cache_resource(ttl=3600, show_spinner="Building time series…")
def panel(src: tuple, freq: str) -> pd.DataFrame:
    sales, promos, _ = load(src)
    return an.build_panel(sales, promos, an.Settings(freq=freq))


@st.cache_data(ttl=3600, show_spinner="Analysing promotions…")
def promo_result(
    src, freq, by_location, max_lag, window, min_events, low, high
) -> an.Result:
    sales, promos, _ = load(src)
    return an.analyse(
        sales,
        promos,
        an.Settings(
            freq=freq,
            by_location=by_location,
            max_lag=max_lag,
            baseline_window=window,
            min_events=min_events,
            low_dependency=low,
            high_dependency=high,
        ),
    )


@st.cache_data(ttl=3600, show_spinner="Analysing seasonality…")
def season_result(src) -> dict:
    return se.analyse(panel(src, "D"), panel(src, "W-MON"))


@st.cache_data(ttl=3600, show_spinner="Analysing holidays…")
def holiday_result(src, country: str, product: str | None) -> pd.DataFrame:
    d = panel(src, "D")
    if product:
        d = d[d["product"] == product]
    return ce.holiday_effects(d.groupby("period")["qty"].sum(), country)


@st.cache_data(ttl=3600, show_spinner="Checking data quality…")
def dq_result(src, country: str) -> dict:
    cid, start, _, demo = src
    sales, _, _ = load(src)
    if demo:
        from demo import make_demo_quality

        quality, today = make_demo_quality(sales), sales["date"].max() + pd.Timedelta(
            days=1
        )
    else:
        import queries as q

        today = pd.Timestamp.now(tz=q.LOCAL_TZ).tz_localize(None)
        try:
            quality = q.load_quality(cid, start)
        except (
            Exception
        ) as err:  # e.g. missing permissions: show everything that doesn't need it
            st.warning(f"Arrival and change statistics are unavailable: {err}")
            quality = None
    years = list(range(sales["date"].min().year, sales["date"].max().year + 1))
    holidays = ce.holiday_calendar(country, years)["date"]
    return dq.analyse(sales, panel(src, "D"), quality, holidays, today)


@st.cache_data(ttl=3600, show_spinner="Backtesting benchmark forecasts…")
def bench_result(src, freq: str, horizon: int, test_periods: int) -> dict:
    return bm.backtest(panel(src, freq), freq, horizon, test_periods)
