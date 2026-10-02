"""Portfolio analytics for forecasting customers.  Run with:  streamlit run app.py

Sidebar, data loading and tab layout. Each tab lives in views/<tab>.py; article mode
(one articleId, no customer needed) is views/article.py.
"""

from __future__ import annotations

import uuid
from datetime import date

import streamlit as st

import calendar_effects as ce
from views import article, explorer, forecast, holiday, overview, promo, quality, season
from views.common import WINDOWS, ArticleContext, Context
from views.data import (
    article_result,
    bench_result,
    dq_result,
    holiday_result,
    load,
    load_article,
    promo_result,
    season_result,
)

st.set_page_config(page_title="Portfolio analytics", page_icon="📈", layout="wide")

# freq, unit, promo baseline window, promo max lag, benchmark horizon, benchmark test periods
GRANULARITY = {
    "Weekly": ("W-MON", "week", 13, 3, 1, 26),
    "Daily": ("D", "day", 35, 14, 2, 91),
}

# ---------------------------------------------------------------------- sidebar

DEMO_ARTICLE = "009705"  # has a data-entry spike and stops with its account

with st.sidebar:
    st.header("Configuration")
    # outside the form, so the fields below switch as soon as the mode does
    mode = st.segmented_control(
        "Analyse",
        ["Customer", "Article"],
        default="Article" if "article" in st.query_params else "Customer",
        required=True,
    )
    with st.form("params"):
        demo = st.toggle(
            "Use demo data",
            value=False,
            help="Synthetic data with known effects, no DB needed.",
        )
        if mode == "Article":
            article_id = st.text_input(
                "articleId",
                value=st.query_params.get("article", ""),
                help=f"Internal article ID; no customer needed. Demo data: e.g. {DEMO_ARTICLE}.",
            )
        else:
            cid = st.text_input("CID", value=st.query_params.get("cid", ""))
        start = st.date_input("History from", value=date(2022, 1, 1))
        granularity = st.radio(
            "Granularity",
            list(GRANULARITY),
            horizontal=True,
            help="Used for promotions and benchmarks. Seasonality and holidays "
            "always use daily and weekly data.",
        )
        country = st.selectbox(
            "Holiday calendar", list(ce.COUNTRIES), format_func=ce.COUNTRIES.get
        )
        with st.expander("Promotion settings"):
            promo_window = st.radio(
                "Promo dates to test against",
                list(WINDOWS),
                format_func=WINDOWS.get,
                help="For producer customers the primary window is the ordering "
                "period and the secondary window is the shelf promo at the retailer.",
            )
            by_location = mode == "Customer" and st.checkbox(
                "Split by account",
                help="Needed when stores and distributors behave differently.",
            )
            max_lag = st.slider("Largest shift to test (periods, 0 = auto)", 0, 28, 0)
            window = st.number_input("Baseline window (periods, 0 = auto)", 0, 104, 0)
            min_events = st.slider(
                "Promos needed to trust an article's shift", 1, 10, 3
            )
            low, high = (
                st.slider("Dependency bands (incremental share)", 0.0, 1.0, (0.10, 0.30))
                if mode == "Customer"
                else (0.10, 0.30)
            )
        st.form_submit_button("Run analysis", type="primary", width="stretch")

freq, unit, default_window, default_lag, default_h, default_test = GRANULARITY[
    granularity
]
max_lag = max_lag or default_lag

# ------------------------------------------------------------------ article mode

if mode == "Article":
    article_id = article_id.strip() or (DEMO_ARTICLE if demo else "")
    if not article_id:
        st.title("Portfolio analytics")
        st.info("Enter an articleId in the sidebar, or switch on demo data to try the tool.")
        st.stop()
    if not demo:
        st.query_params.clear()
        st.query_params["article"] = article_id  # makes the URL shareable
    asrc = (article_id, start, promo_window, demo)
    a_sales, _, a_info = load_article(asrc)
    if a_sales.empty:
        st.title("Portfolio analytics")
        st.warning(
            f"No sales for articleId {article_id} since the selected date. "
            "Check the ID or move the start date back."
        )
        st.stop()
    article.render(
        ArticleContext(
            asrc=asrc,
            article_id=article_id,
            demo=demo,
            promo_window=promo_window,
            info=a_info,
            freq=freq,
            unit=unit,
            max_lag=max_lag,
            sales=a_sales,
            R=article_result(
                asrc, freq, country, max_lag, window or default_window, min_events
            ),
        )
    )
    st.stop()

# ----------------------------------------------------------------- customer mode

if not demo:
    try:
        cid = str(uuid.UUID(cid.strip()))
    except ValueError:
        st.title("Portfolio analytics")
        st.info(
            "Enter a customer ID in the sidebar, or switch on demo data to try the tool."
        )
        st.stop()
    st.query_params.clear()
    st.query_params["cid"] = cid  # makes the URL shareable

src = (cid, start, promo_window, demo)
sales, _, info = load(src)
if sales.empty:
    st.title("Portfolio analytics")
    st.warning(
        "No sales for this customer since the selected date. Check the customer ID or move the start date back."
    )
    st.stop()

horizon = st.session_state.get(f"h_{freq}", default_h)
test_periods = st.session_state.get(f"test_{freq}", default_test)

S = season_result(src)
B = bench_result(src, freq, horizon, test_periods)
P = promo_result(
    src, freq, by_location, max_lag, window or default_window, min_events, low, high
)
H = holiday_result(src, country, None)
Q = dq_result(src, country)

st.title("Portfolio analytics")
st.caption(
    "Demo data"
    if demo
    else f"Customer {cid}. {len(sales):,} sales rows from {sales['date'].min():%d %b %Y} "
    f"to {sales['date'].max():%d %b %Y}, {info['promo_rows']:,} promotion rows."
)

ctx = Context(
    src=src,
    cid=cid,
    demo=demo,
    country=country,
    promo_window=promo_window,
    info=info,
    freq=freq,
    unit=unit,
    horizon=horizon,
    test_periods=test_periods,
    default_h=default_h,
    default_test=default_test,
    by_location=by_location,
    max_lag=max_lag,
    min_events=min_events,
    low=low,
    high=high,
    S=S,
    B=B,
    P=P,
    H=H,
    Q=Q,
)

TABS = {
    "Overview": overview,
    "Data quality": quality,
    "Seasonality": season,
    "Holidays": holiday,
    "Forecastability": forecast,
    "Promotions": promo,
    "Article explorer": explorer,
}
for tab, view in zip(st.tabs(list(TABS)), TABS.values()):
    with tab:
        view.render(ctx)
