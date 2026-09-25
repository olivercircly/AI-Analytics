"""Portfolio analytics for forecasting customers.  Run with:  streamlit run app.py"""

from __future__ import annotations

import uuid
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import analysis as an
import benchmark as bm
import calendar_effects as ce
import seasonality as se

st.set_page_config(page_title="Portfolio analytics", page_icon="📈", layout="wide")

# freq, unit, promo baseline window, promo max lag, benchmark horizon, benchmark test periods
GRANULARITY = {
    "Weekly": ("W-MON", "week", 13, 3, 1, 26),
    "Daily": ("D", "day", 35, 14, 2, 91),
}
WINDOWS = {
    "primary": "Primary (validFrom–validTo)",
    "secondary": "Secondary (retail promo period)",
}
RED, ORANGE, BLUE, GREY, INK = "#c0392b", "#e67e22", "#2980b9", "#95a5a6", "#2c3e50"


def pct(x: float, signed: bool = False) -> str:
    return "–" if x is None or pd.isna(x) else (f"{x:+.0%}" if signed else f"{x:.0%}")


# ------------------------------------------------------------------ data + cache
# src = (cid, start, promo_window, demo) identifies the data; everything is cached on it.


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


@st.cache_data(ttl=3600, show_spinner="Backtesting benchmark forecasts…")
def bench_result(src, freq: str, horizon: int, test_periods: int) -> dict:
    return bm.backtest(panel(src, freq), freq, horizon, test_periods)


# ---------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Customer")
    with st.form("params"):
        demo = st.toggle(
            "Use demo data",
            value=False,
            help="Synthetic data with known effects, no DB needed.",
        )
        cid = st.text_input("Customer ID (cid)", value=st.query_params.get("cid", ""))
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
            by_location = st.checkbox(
                "Split by account",
                help="Needed when stores and distributors behave differently.",
            )
            max_lag = st.slider("Largest shift to test (periods, 0 = auto)", 0, 28, 0)
            window = st.number_input("Baseline window (periods, 0 = auto)", 0, 104, 0)
            min_events = st.slider(
                "Promos needed to trust an article's shift", 1, 10, 3
            )
            low, high = st.slider(
                "Dependency bands (incremental share)", 0.0, 1.0, (0.10, 0.30)
            )
        st.form_submit_button("Run analysis", type="primary", width="stretch")

freq, unit, default_window, default_lag, default_h, default_test = GRANULARITY[
    granularity
]
max_lag = max_lag or default_lag
if not demo:
    try:
        cid = str(uuid.UUID(cid.strip()))
    except ValueError:
        st.title("Portfolio analytics")
        st.info(
            "Enter a customer ID in the sidebar, or switch on demo data to try the tool."
        )
        st.stop()
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

st.title("Portfolio analytics")
st.caption(
    "Demo data"
    if demo
    else f"Customer {cid}. {len(sales):,} sales rows from {sales['date'].min():%d %b %Y} "
    f"to {sales['date'].max():%d %b %Y}, {info['promo_rows']:,} promotion rows."
)

tabs = st.tabs(
    [
        "Overview",
        "Seasonality",
        "Holidays",
        "Forecastability",
        "Promotions",
        "Article explorer",
    ]
)

# --------------------------------------------------------------------- overview

with tabs[0]:
    weekly_total = panel(src, "W-MON").groupby("period")["qty"].sum()
    last52 = weekly_total.iloc[-52:].sum()
    yoy = (
        last52 / weekly_total.iloc[-104:-52].sum() - 1
        if len(weekly_total) >= 104
        else np.nan
    )
    vol = S["per_article"].set_index("product")["volume"].sort_values(ascending=False)
    n80 = int((vol.cumsum() / vol.sum() < 0.8).sum() + 1)
    recent = panel(src, "D")
    active = recent.loc[
        recent["period"] >= recent["period"].max() - pd.Timedelta(days=90)
    ]
    active = active.groupby("product")["qty"].sum().gt(0).sum()

    sp, bp, pp = S["portfolio"], B["portfolio"], P.portfolio
    c = st.columns(6)
    c[0].metric(
        "Articles",
        f"{len(vol):,}",
        f"{active:,} sold in last 90 days",
        delta_color="off",
    )
    c[1].metric(
        "Volume, last 52 weeks",
        f"{last52:,.0f}",
        pct(yoy, signed=True) + " vs. year before" if pd.notna(yoy) else None,
    )
    c[2].metric(
        "Weekday seasonality", se.classify(pd.Series([sp["weekday_strength"]])).iloc[0]
    )
    c[3].metric(
        "Yearly seasonality",
        se.classify(pd.Series([sp["yearly_strength"]])).iloc[0],
        f"peak in {sp['peak_month']}" if sp["peak_month"] else None,
        delta_color="off",
    )
    c[4].metric(
        "Benchmark error (WAPE)",
        pct(bp["wape"]),
        f"{bp['best_method']}",
        delta_color="off",
        help=f"Best simple method, per article and {unit}, {horizon} {unit}(s) ahead.",
    )
    c[5].metric(
        "Promo uplift", pct(pp["incremental_share"]), "of volume", delta_color="off"
    )

    findings = [
        f"**Concentration:** {n80:,} of {len(vol):,} articles make up 80% of volume."
    ]
    wp = S["weekday_profile"]
    if wp is not None and wp["index"].notna().any():
        closed = wp.loc[wp["index"] < 0.05, "weekday"].tolist()
        open_days = wp[wp["index"] >= 0.05]
        busiest, quietest = (
            open_days.loc[open_days["index"].idxmax()],
            open_days.loc[open_days["index"].idxmin()],
        )
        findings.append(
            f"**Week:** busiest day is {busiest['weekday']} ({pct(busiest['index'] - 1, True)} vs. average), "
            f"quietest open day is {quietest['weekday']} ({pct(quietest['index'] - 1, True)})"
            + (f"; no sales on {', '.join(closed)}." if closed else ".")
        )
    too_short = (S["per_article"]["yearly_class"] == "too short").sum()
    findings.append(
        f"**Year:** {pct(sp['seasonal_yearly_volume'])} of volume is in articles with at least moderate "
        f"yearly seasonality"
        + (
            f", peaking in {sp['peak_month']} for the portfolio"
            if sp["peak_month"]
            else ""
        )
        + (
            f". {too_short:,} articles have under two years of history, too short to tell."
            if too_short
            else "."
        )
    )
    hs = ce.summarise(H)
    if hs["closed"] or hs["biggest_uplift"]:
        txt = (
            f"**Holidays:** closed on {len(hs['closed'])} holidays"
            if hs["closed"]
            else "**Holidays:** open on all holidays"
        )
        if hs["biggest_uplift"]:
            name, off, eff = hs["biggest_uplift"]
            txt += (
                f"; biggest uplift is {pct(eff, True)} {abs(off)} day{'s' if abs(off) > 1 else ''} "
                f"{'before' if off < 0 else 'after'} {name}"
            )
        findings.append(txt + ".")
    hard = B["articles"].loc[
        B["articles"]["forecastability"] == "hard", "test_volume"
    ].sum() / max(B["articles"]["test_volume"].sum(), 1)
    findings.append(
        f"**Forecastability:** the best simple method ({bp['best_method']}) is off by {pct(bp['wape'])} per article "
        f"but only {pct(bp['total_wape'])} on the portfolio total. {pct(hard)} of volume is in hard articles "
        f"(error above 50%). Promotions cover {pct(bp['promo_period_share'])} of test periods but cause "
        f"{pct(bp['promo_error_share'])} of the error."
    )
    shifted = P.metrics[P.metrics["best_lag"].fillna(0) != 0]
    findings.append(
        f"**Promotions:** {pct(pp['incremental_share'])} of volume is promo uplift"
        + (
            f"; {len(shifted)} articles peak away from the recorded promo dates (see Promotions)."
            if len(shifted)
            else "."
        )
    )
    st.markdown("\n".join(f"- {f}" for f in findings))

    st.altair_chart(
        alt.Chart(weekly_total.rename("units").reset_index())
        .mark_line(color=INK)
        .encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("units:Q", title="Units per week"),
            tooltip=[
                alt.Tooltip("period:T", title="Week of"),
                alt.Tooltip("units:Q", format=",.0f"),
            ],
        )
        .properties(height=260),
        width="stretch",
    )

# ------------------------------------------------------------------ seasonality

with tabs[1]:
    pa = S["per_article"]
    c = st.columns(4)
    c[0].metric(
        "Weekday strength",
        f"{sp['weekday_strength']:.2f}" if pd.notna(sp["weekday_strength"]) else "–",
        help="0 = no weekday pattern, 1 = sales fully explained by the weekday. Portfolio total.",
    )
    c[1].metric(
        "Yearly strength",
        (
            f"{sp['yearly_strength']:.2f}"
            if pd.notna(sp["yearly_strength"])
            else "too short"
        ),
        help="Same scale for the week of the year. Needs two years of history.",
    )
    c[2].metric(
        "Peak-to-trough",
        pct(sp["amplitude"]),
        f"peak in {sp['peak_month']}" if sp["peak_month"] else None,
        delta_color="off",
    )
    c[3].metric(
        "Volume with strong yearly seasonality", pct(sp["strong_yearly_volume"])
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Average week")
        if wp is not None:
            st.altair_chart(
                alt.Chart(wp)
                .mark_bar(color=BLUE)
                .encode(
                    x=alt.X("weekday:N", sort=se.WEEKDAYS, title=None),
                    y=alt.Y(
                        "index:Q",
                        title="Sales vs. daily average",
                        axis=alt.Axis(format="%"),
                    ),
                    tooltip=["weekday", alt.Tooltip("index:Q", format=".0%")],
                ),
                width="stretch",
            )
    with right:
        st.subheader("Year over year")
        st.altair_chart(
            alt.Chart(S["overlay"])
            .mark_line()
            .encode(
                x=alt.X(
                    "week:Q", title="Week of year", scale=alt.Scale(domain=[1, 53])
                ),
                y=alt.Y("qty:Q", title="Units per week"),
                color=alt.Color("year:O", title=None),
                tooltip=["year", "week", alt.Tooltip("qty:Q", format=",.0f")],
            ),
            width="stretch",
        )

    left, right = st.columns(2)
    with left:
        st.subheader("When do seasonal articles peak?")
        seasonal = pa[pa["yearly_class"].isin(["strong", "moderate"])]
        if seasonal.empty:
            st.info("No article has two years of history with a clear yearly pattern.")
        else:
            peaks = (
                seasonal.groupby("peak_month")
                .agg(articles=("product", "size"), volume=("volume", "sum"))
                .reset_index()
            )
            st.altair_chart(
                alt.Chart(peaks)
                .mark_bar(color=ORANGE)
                .encode(
                    x=alt.X(
                        "peak_month:N",
                        sort=se.MONTHS,
                        title=None,
                        scale=alt.Scale(domain=se.MONTHS),
                    ),
                    y=alt.Y("volume:Q", title="Volume of articles peaking that month"),
                    tooltip=[
                        "peak_month",
                        "articles",
                        alt.Tooltip("volume:Q", format=",.0f"),
                    ],
                ),
                width="stretch",
            )
    with right:
        st.subheader("Weekday vs. yearly pattern per article")
        st.altair_chart(
            alt.Chart(pa.dropna(subset=["weekday_strength", "yearly_strength"]))
            .mark_circle(opacity=0.6, color=INK)
            .encode(
                x=alt.X(
                    "weekday_strength:Q",
                    title="Weekday strength",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    "yearly_strength:Q",
                    title="Yearly strength",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                size=alt.Size("volume:Q", legend=None),
                tooltip=[
                    "product",
                    alt.Tooltip("volume:Q", format=",.0f"),
                    alt.Tooltip("weekday_strength:Q", format=".2f"),
                    alt.Tooltip("yearly_strength:Q", format=".2f"),
                    "peak_month",
                ],
            )
            .interactive(),
            width="stretch",
        )

    st.dataframe(
        pa,
        hide_index=True,
        width="stretch",
        column_config={
            "product": "Article",
            "volume": st.column_config.NumberColumn("Volume", format="%,.0f"),
            "weekday_strength": st.column_config.ProgressColumn(
                "Weekday strength", format="%.2f", min_value=0, max_value=1
            ),
            "yearly_strength": st.column_config.ProgressColumn(
                "Yearly strength", format="%.2f", min_value=0, max_value=1
            ),
            "weekday_class": "Weekday",
            "yearly_class": "Yearly",
            "amplitude": st.column_config.NumberColumn(
                "Peak-to-trough", format="percent"
            ),
            "peak_month": "Peak month",
        },
    )
    st.caption(
        "Promotion periods are excluded, so promos don't show up as seasonality. "
        "Strength: 0–0.3 weak, 0.3–0.6 moderate, above 0.6 strong."
    )

# --------------------------------------------------------------------- holidays

with tabs[2]:
    top = S["per_article"]["product"].head(200).tolist()
    article = st.selectbox(
        "Show effects for",
        ["Whole portfolio"] + top,
        key="hol_article",
        help="Articles sorted by volume (top 200).",
    )
    effects = (
        H if article == "Whole portfolio" else holiday_result(src, country, article)
    )
    if effects.empty or effects["effect"].isna().all():
        st.info(
            "Not enough history around holidays yet (needs five weeks before and after each one)."
        )
    else:
        hs = ce.summarise(effects)
        notes = []
        if hs["closed"]:
            notes.append(
                f"No sales on {len(hs['closed'])} days: {', '.join(hs['closed'])}."
            )
        if hs["biggest_uplift"]:
            name, off, eff = hs["biggest_uplift"]
            notes.append(
                f"Biggest uplift: {pct(eff, True)} {abs(off)} day{'s' if abs(off) > 1 else ''} "
                f"{'before' if off < 0 else 'after'} {name}."
            )
        if hs["biggest_drop"]:
            notes.append(
                f"Biggest drop on an open holiday: {hs['biggest_drop'][0]} ({pct(hs['biggest_drop'][1], True)})."
            )
        st.markdown(" ".join(notes))

        eff = effects.assign(
            label=effects["effect"].map(lambda v: "" if pd.isna(v) else f"{v:+.0%}"),
            strong=effects["effect"].abs() > 0.5,
        )
        base = alt.Chart(eff).encode(
            x=alt.X("offset:O", title="Days relative to the holiday"),
            y=alt.Y(
                "holiday:N", title=None, sort=alt.EncodingSortField("order", op="min")
            ),
        )
        heat = base.mark_rect().encode(
            color=alt.Color(
                "effect:Q",
                title="vs. normal",
                legend=alt.Legend(format="%"),
                scale=alt.Scale(scheme="redblue", domain=[-1, 1], clamp=True),
            ),
            tooltip=[
                "holiday",
                "offset",
                alt.Tooltip("effect:Q", format="+.0%"),
                "occurrences",
            ],
        )
        text = base.mark_text(fontSize=11).encode(
            text="label:N",
            color=alt.condition(alt.datum.strong, alt.value("white"), alt.value(INK)),
        )
        st.altair_chart(
            (heat + text).properties(height=28 * effects["holiday"].nunique() + 40),
            width="stretch",
        )
        st.caption(
            f"{ce.COUNTRIES[country]} public holidays plus Easter and Pentecost Sunday, Christmas Eve, "
            "New Year's Eve and bridge days. Normal = median of the same weekday 1–4 weeks before and "
            "after. Holidays are sorted by date in the year. School holidays aren't included."
        )

# --------------------------------------------------------------- forecastability

with tabs[3]:
    c = st.columns([1, 1, 2])
    c[0].number_input(
        f"Forecast horizon ({unit}s ahead)",
        1,
        60 if freq == "D" else 26,
        default_h,
        key=f"h_{freq}",
    )
    c[1].number_input(
        f"Test window ({unit}s)",
        4,
        365 if freq == "D" else 104,
        default_test,
        key=f"test_{freq}",
    )
    arts = B["articles"]
    c = st.columns(4)
    c[0].metric("Best simple method", bp["best_method"])
    c[1].metric(
        f"Error per article-{unit}",
        pct(bp["wape"]),
        help="WAPE: absolute error divided by actual volume.",
    )
    c[2].metric(
        "Error on portfolio total",
        pct(bp["total_wape"]),
        help="Errors partly cancel when articles are summed; this is the accuracy of the total.",
    )
    c[3].metric("Bias", pct(bp["bias"], True), help="Positive = over-forecasting.")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Error by method")
        st.altair_chart(
            alt.Chart(B["by_method"])
            .mark_bar()
            .encode(
                y=alt.Y("method:N", sort="x", title=None),
                x=alt.X("wape:Q", title="WAPE", axis=alt.Axis(format="%")),
                color=alt.condition(
                    alt.datum.method == bp["best_method"],
                    alt.value(BLUE),
                    alt.value(GREY),
                ),
                tooltip=[
                    "method",
                    alt.Tooltip("wape:Q", format=".1%"),
                    alt.Tooltip("bias:Q", format="+.1%"),
                ],
            ),
            width="stretch",
        )
        share = arts.groupby("forecastability")["test_volume"].sum() / max(
            arts["test_volume"].sum(), 1
        )
        st.markdown(
            " · ".join(
                f"**{k}** {pct(share.get(k, 0))} of volume"
                for k in ["easy", "medium", "hard"]
            )
        )
        st.markdown(
            f"Promotions: {pct(bp['promo_period_share'])} of test periods, "
            f"{pct(bp['promo_error_share'])} of the error."
        )
    with right:
        st.subheader("Small articles are harder to forecast")
        st.altair_chart(
            alt.Chart(arts[arts["test_volume"] > 0])
            .mark_circle(opacity=0.6)
            .encode(
                x=alt.X(
                    "test_volume:Q",
                    title="Volume in test window",
                    scale=alt.Scale(type="log"),
                ),
                y=alt.Y("best_wape:Q", title="Best WAPE", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "forecastability:N",
                    scale=alt.Scale(
                        domain=["easy", "medium", "hard"], range=[BLUE, ORANGE, RED]
                    ),
                ),
                tooltip=[
                    "product",
                    "best_method",
                    alt.Tooltip("best_wape:Q", format=".0%"),
                    alt.Tooltip("test_volume:Q", format=",.0f"),
                ],
            )
            .interactive(),
            width="stretch",
        )

    st.dataframe(
        arts,
        hide_index=True,
        width="stretch",
        column_config={
            "product": "Article",
            "test_volume": st.column_config.NumberColumn("Test volume", format="%,.0f"),
            "best_method": "Best method",
            "best_wape": st.column_config.NumberColumn("Best WAPE", format="percent"),
            "portfolio_method_wape": st.column_config.NumberColumn(
                f"WAPE with {bp['best_method']}", format="percent"
            ),
            "forecastability": "Forecastability",
        },
    )
    st.caption(
        f"Each method forecasts every {unit} of the last {test_periods} {unit}s, {horizon} {unit}(s) ahead, "
        "using only data available at the time. Easy: WAPE below 20%, hard: above 50%. "
        "Your production model should beat the best of these."
    )

# ------------------------------------------------------------------- promotions

with tabs[4]:
    if not demo:
        st.caption(
            f"Testing the {WINDOWS[promo_window].lower()}. {info['secondary_rows']:,} of "
            f"{info['promo_rows']:,} promotion rows have a secondary window."
        )
        if promo_window == "secondary" and info["secondary_rows"] == 0:
            st.warning(
                "This customer has no secondary promo windows. Switch to the primary window."
            )
    m = P.metrics
    sub = st.tabs(["Summary", "Dependency", "Shift"])
    with sub[0]:
        c = st.columns(5)
        c[0].metric(
            "Articles with promos",
            f"{pp['promoted_series']:,}",
            f"of {pp['series']:,}",
            delta_color="off",
        )
        c[1].metric(
            "Sold during promos",
            pct(pp["promo_volume_share"]),
            f"{pct(pp['promo_volume_share_adj'] - pp['promo_volume_share'], True)} after shift correction",
            delta_color="off",
        )
        c[2].metric(
            "Incremental volume",
            pct(pp["incremental_share"]),
            help="Volume above baseline during (shift-adjusted) promos, as a share of all volume.",
        )
        c[3].metric(
            "Volume in high-dependency articles", pct(pp["high_dependency_volume"])
        )
        c[4].metric(
            "Shifted volume",
            pct(pp["shifted_volume_share"]),
            help=f"Share of volume in articles whose sales peak before or after recorded promos "
            f"({pp['tested_series']} articles had enough promos to test).",
        )
        inc = pp["incremental_share"]
        notes = [
            f"The portfolio is **{'strongly' if inc >= high else 'moderately' if inc >= low else 'barely'} "
            f"promo-driven**: {pct(inc)} of volume is promo uplift."
        ]
        if len(shifted):
            typical = shifted["best_lag"].mode().iloc[0]
            n = abs(int(typical))
            notes.append(
                f"**{len(shifted)} articles** peak away from the recorded promo dates, most often "
                f"**{n} {unit}{'s' if n > 1 else ''} {'before' if typical < 0 else 'after'}**. Sales before "
                "a promo usually mean sell-in (a distributor stocking up for a store promotion); sales after "
                "usually mean announcement dates or lagging data. Shift promo features by that lag."
            )
            if shifted["best_lag"].abs().eq(max_lag).any():
                notes.append(
                    f"⚠️ Some shifts sit at the edge of the tested range (±{max_lag} {unit}s). "
                    "Increase *Largest shift to test* in Promotion settings."
                )
        else:
            notes.append("Promo effects line up with the recorded promo dates.")
        st.markdown("\n\n".join(notes))
        st.altair_chart(
            alt.Chart(P.pooled_lag_scores.reset_index())
            .mark_bar()
            .encode(
                x=alt.X(
                    "lag:O",
                    title=f"Sales timing vs. recorded promo ({unit}s, negative = before)",
                ),
                y=alt.Y("score:Q", title="Uplift during promo vs. outside"),
                color=alt.condition(
                    alt.datum.lag == pp["shift"], alt.value(RED), alt.value(GREY)
                ),
                tooltip=["lag", alt.Tooltip("score:Q", format=".2f")],
            ),
            width="stretch",
        )
    with sub[1]:
        st.altair_chart(
            alt.Chart(m.dropna(subset=["incremental_share"]))
            .mark_circle(opacity=0.7)
            .encode(
                x=alt.X(
                    "promo_period_share:Q",
                    title="Share of time on promo",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y(
                    "incremental_share:Q",
                    title="Incremental share of volume",
                    axis=alt.Axis(format="%"),
                ),
                size=alt.Size("volume:Q", legend=None),
                color=alt.Color(
                    "dependency:N",
                    scale=alt.Scale(
                        domain=["high", "medium", "low", "no promos"],
                        range=[RED, ORANGE, BLUE, "#bdc3c7"],
                    ),
                ),
                tooltip=[
                    "series",
                    alt.Tooltip("volume:Q", format=",.0f"),
                    "promo_events",
                    alt.Tooltip("incremental_share:Q", format=".0%"),
                    alt.Tooltip("promo_lift:Q", format=".2f"),
                ],
            )
            .interactive(),
            width="stretch",
        )
        bar = lambda label: st.column_config.ProgressColumn(
            label, format="percent", min_value=0, max_value=1
        )
        st.dataframe(
            m[
                [
                    "series",
                    "dependency",
                    "volume",
                    "promo_events",
                    "promo_period_share",
                    "promo_volume_share",
                    "promo_volume_share_adj",
                    "incremental_share",
                    "promo_lift",
                    "best_lag",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "series": "Article",
                "dependency": "Dependency",
                "volume": st.column_config.NumberColumn("Volume", format="%,.0f"),
                "promo_events": "Promos",
                "promo_period_share": bar("Time on promo"),
                "promo_volume_share": bar("Sold on promo"),
                "promo_volume_share_adj": bar("Sold on promo (shift-adj.)"),
                "incremental_share": bar("Incremental"),
                "promo_lift": st.column_config.NumberColumn("Lift ×", format="%.2f"),
                "best_lag": st.column_config.NumberColumn(
                    f"Shift ({unit}s)", format="%+d"
                ),
            },
        )
        st.download_button(
            "Download table as CSV",
            m.to_csv(index=False).encode(),
            file_name=f"promo_metrics_{cid or 'demo'}.csv",
            mime="text/csv",
        )
    with sub[2]:
        left, right = st.columns(2)
        with left:
            st.subheader("Average sales around a promo start")
            prof = P.event_profile
            if prof.empty:
                st.info("No isolated promo starts to average.")
            else:
                x = alt.X(
                    "offset:Q", title=f"{unit.title()}s from recorded promo start"
                )
                st.altair_chart(
                    alt.Chart(prof)
                    .mark_area(opacity=0.2, color=BLUE)
                    .encode(x=x, y="lo:Q", y2="hi:Q")
                    + alt.Chart(prof)
                    .mark_line(point=True, color=BLUE)
                    .encode(
                        x=x,
                        y=alt.Y("mean:Q", title="Uplift (× average sales)"),
                        tooltip=[
                            "offset",
                            alt.Tooltip("mean:Q", format=".2f"),
                            "events",
                        ],
                    )
                    + alt.Chart(pd.DataFrame({"x": [0]}))
                    .mark_rule(strokeDash=[4, 4])
                    .encode(x="x:Q"),
                    width="stretch",
                )
                st.caption(
                    f"Based on {int(prof['events'].max())} promos with no other promo just before. "
                    "Uplift before 0 means sales anticipate the promo; a dip after means pull-forward."
                )
        with right:
            st.subheader("Detected shift per article")
            dist = (
                m["best_lag"]
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
                .rename_axis("lag")
                .reset_index(name="articles")
            )
            if dist.empty:
                st.info(f"No article has {min_events}+ promos yet.")
            else:
                st.altair_chart(
                    alt.Chart(dist)
                    .mark_bar(color=GREY)
                    .encode(
                        x=alt.X(
                            "lag:O",
                            title=f"Shift ({unit}s, negative = sales before promo)",
                        ),
                        y=alt.Y("articles:Q", title="Articles"),
                        tooltip=["lag", "articles"],
                    ),
                    width="stretch",
                )
                if by_location:
                    st.dataframe(
                        m.dropna(subset=["best_lag"])
                        .groupby("location")
                        .agg(
                            articles=("series", "size"),
                            median_shift=("best_lag", "median"),
                            volume=("volume", "sum"),
                        )
                        .sort_values("volume", ascending=False),
                        width="stretch",
                    )
                    st.caption(
                        "Accounts with a consistent negative shift are likely distributors or DCs."
                    )

# -------------------------------------------------------------- article explorer

with tabs[5]:
    pick = st.selectbox("Article (sorted by volume)", m["series"], key="explorer_pick")
    row = m.set_index("series").loc[pick]
    product = row["product"]
    srow = S["per_article"].set_index("product").reindex([product]).iloc[0]
    brow = B["articles"].set_index("product").reindex([product]).iloc[0]
    c = st.columns(6)
    c[0].metric("Weekday", srow["weekday_class"])
    c[1].metric(
        "Yearly",
        srow["yearly_class"],
        f"peak in {srow['peak_month']}" if pd.notna(srow["peak_month"]) else None,
        delta_color="off",
    )
    c[2].metric(
        "Benchmark WAPE",
        pct(brow["best_wape"]),
        brow["best_method"] if pd.notna(brow["best_method"]) else None,
        delta_color="off",
    )
    c[3].metric("Promos", int(row["promo_events"]))
    c[4].metric("Promo uplift", pct(row["incremental_share"]))
    lag = row["best_lag"]
    c[5].metric(
        "Promo shift",
        (
            "aligned"
            if pd.isna(lag) or lag == 0
            else f"{abs(int(lag))} {unit}{'s' if abs(lag) > 1 else ''} {'before' if lag < 0 else 'after'}"
        ),
    )

    d = P.panel[P.panel["series"] == pick].assign(
        period_end=lambda x: x["period"] + P.settings.period_length
    )
    x = alt.X("period:T", title=None)
    bands = (
        alt.Chart(d[d["promo"]])
        .mark_rect(color="#f39c12", opacity=0.25)
        .encode(x=x, x2="period_end:T")
    )
    lines = (
        alt.Chart(d)
        .transform_fold(["qty", "baseline"], as_=["measure", "value"])
        .mark_line()
        .encode(
            x=x,
            y=alt.Y("value:Q", title="Units"),
            color=alt.Color(
                "measure:N",
                scale=alt.Scale(domain=["qty", "baseline"], range=[INK, GREY]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["period:T", alt.Tooltip("value:Q", format=",.0f"), "measure:N"],
        )
    )
    st.altair_chart(
        (bands + lines).properties(height=340).interactive(bind_y=False),
        width="stretch",
    )
    st.caption("Orange bands are recorded promo periods.")

    left, right = st.columns(2)
    wi, yi = S["weekday_index"], S["yearly_index"]
    with left:
        if product in wi.index:
            st.altair_chart(
                alt.Chart(
                    pd.DataFrame(
                        {"weekday": se.WEEKDAYS, "index": wi.loc[product].to_numpy()}
                    )
                )
                .mark_bar(color=BLUE)
                .encode(
                    x=alt.X("weekday:N", sort=se.WEEKDAYS, title=None),
                    y=alt.Y(
                        "index:Q",
                        title="Sales vs. daily average",
                        axis=alt.Axis(format="%"),
                    ),
                ),
                width="stretch",
            )
    with right:
        if product in yi.index and pd.notna(srow["yearly_strength"]):
            yd = pd.DataFrame(
                {"week": yi.columns, "Article": yi.loc[product].to_numpy()}
            )
            if S["yearly_profile"] is not None:
                yd["Portfolio"] = S["yearly_profile"]["index"].to_numpy()
            st.altair_chart(
                alt.Chart(yd)
                .transform_fold(
                    [c for c in yd.columns if c != "week"], as_=["series", "index"]
                )
                .mark_line()
                .encode(
                    x=alt.X("week:Q", title="Week of year"),
                    y=alt.Y(
                        "index:Q", title="Seasonal index", axis=alt.Axis(format="%")
                    ),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(range=[ORANGE, GREY]),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                ),
                width="stretch",
            )
        else:
            st.info("Under two years of history, so no yearly pattern yet.")
