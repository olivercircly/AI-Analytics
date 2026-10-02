"""Data loading and cached analysis results.

src = (cid, start, promo_window, demo) identifies a customer's data, asrc =
(article_id, start, promo_window, demo) a single article's; everything is cached on them.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import analysis as an
import anomalies as ab
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


# ------------------------------------------------------------------ article mode
# asrc = (article_id, start, promo_window, demo) identifies one article's data.


@st.cache_resource(ttl=3600, show_spinner="Loading article…")
def load_article(asrc: tuple) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    article_id, start, promo_window, demo = asrc
    if demo:
        from demo import make_demo_data

        sales, promos = make_demo_data()
        end = sales["date"].max()
        sales = sales[(sales["product"] == article_id) & (sales["qty"] != 0)]
        sales = sales[sales["date"] >= pd.Timestamp(start)]
        return (
            sales.assign(article_number=sales["product"]),
            promos[promos["product"] == article_id],
            {"cid": "demo", "promo_rows": int((promos["product"] == article_id).sum()),
             "secondary_rows": 0, "end": end},
        )
    import queries as q

    sales, raw, cid = q.load_article(article_id, start)
    info = {
        "cid": cid,
        "promo_rows": len(raw),
        "secondary_rows": int(raw["secondary_start"].notna().sum()) if len(raw) else 0,
        "end": pd.Timestamp.now(tz=q.LOCAL_TZ).tz_localize(None).normalize(),
    }
    promos = (
        q.select_window(raw, promo_window)
        if len(raw)
        else pd.DataFrame(columns=["product", "location", "promo_start", "promo_end"])
    )
    return sales, promos, info


@st.cache_data(ttl=3600, show_spinner="Looking for anomalies…")
def article_result(asrc, freq, country, max_lag, window, min_events) -> dict:
    sales, promos, info = load_article(asrc)
    years = list(range(sales["date"].min().year, sales["date"].max().year + 1))
    holidays = ce.holiday_calendar(country, years)["date"]
    s = an.Settings(freq=freq, max_lag=max_lag, baseline_window=window, min_events=min_events)
    return ab.analyse(sales, promos, s, holidays, end=info["end"])
