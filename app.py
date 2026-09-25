"""Promo analytics for forecasting customers.  Run with:  streamlit run app.py"""

from __future__ import annotations

import uuid
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import analysis as an

st.set_page_config(page_title="Promo analytics", page_icon="📈", layout="wide")

GRANULARITY = {"Weekly": ("W-MON", "week", 13, 3), "Daily": ("D", "day", 35, 14)}


# ------------------------------------------------------------------ data + cache


@st.cache_data(ttl=3600, show_spinner="Analysing…")
def run(
    cid: str,
    start: date,
    promo_window: str,
    freq: str,
    by_location: bool,
    max_lag: int,
    window: int,
    min_events: int,
    low: float,
    high: float,
    demo: bool,
) -> tuple[an.Result, dict]:
    if demo:
        from demo import make_demo_data

        sales, promos = make_demo_data()
        sales = sales[sales["date"] >= pd.Timestamp(start)]
        info = {"promo_rows": len(promos), "secondary_rows": 0}
    else:
        import queries as q  # imported lazily so demo mode works without DB credentials

        raw = q.load_promos(cid, start)
        info = {
            "promo_rows": len(raw),
            "secondary_rows": int(raw["secondary_start"].notna().sum()),
        }
        sales, promos = q.label_articles(
            q.load_sales(cid, start), q.select_window(raw, promo_window)
        )
    settings = an.Settings(
        freq=freq,
        by_location=by_location,
        max_lag=max_lag,
        baseline_window=window,
        min_events=min_events,
        low_dependency=low,
        high_dependency=high,
    )
    return an.analyse(sales, promos, settings), info


def describe_lag(k: float, unit: str) -> str:
    if pd.isna(k) or k == 0:
        return "aligned"
    n = abs(int(k))
    return f"{n} {unit}{'s' if n > 1 else ''} {'before' if k < 0 else 'after'}"


# ---------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Customer")
    with st.form("params"):
        demo = st.toggle(
            "Use demo data",
            value=False,
            help="Synthetic data with known shifts, no DB needed.",
        )
        cid = st.text_input("Customer ID (cid)", value=st.query_params.get("cid", ""))
        start = st.date_input("History from", value=date(2022, 1, 1))
        promo_window = st.radio(
            "Promo dates to test against",
            ["primary", "secondary"],
            format_func=lambda w: {
                "primary": "Primary (validFrom–validTo)",
                "secondary": "Secondary (retail promo period)",
            }[w],
            help="For producer customers the primary window is the ordering period and the "
            "secondary window is the shelf promo at the retailer.",
        )
        granularity = st.radio(
            "Granularity",
            list(GRANULARITY),
            horizontal=True,
            help="Weekly is more robust; daily finds shifts shorter than a week.",
        )
        by_location = st.checkbox(
            "Split by location",
            help="Needed when stores and distributors behave differently.",
        )
        with st.expander("Advanced"):
            max_lag = st.slider("Largest shift to test (periods, 0 = auto)", 0, 28, 0)
            window = st.number_input(
                "Baseline window (periods, 0 = default)", 0, 104, 0
            )
            min_events = st.slider("Promos needed to trust a series' shift", 1, 10, 3)
            low, high = st.slider(
                "Dependency bands (incremental share)", 0.0, 1.0, (0.10, 0.30)
            )
        st.form_submit_button("Run analysis", type="primary", width="stretch")

freq, unit, default_window, default_lag = GRANULARITY[granularity]
max_lag = max_lag or default_lag
if not demo:
    try:
        cid = str(uuid.UUID(cid.strip()))
    except ValueError:
        st.title("Promo analytics")
        st.info(
            "Enter a customer ID in the sidebar, or switch on demo data to try the tool."
        )
        st.stop()
    st.query_params["cid"] = cid  # makes the URL shareable

try:
    res, info = run(
        cid,
        start,
        promo_window,
        freq,
        by_location,
        max_lag,
        window or default_window,
        min_events,
        low,
        high,
        demo,
    )
except ValueError as err:
    st.warning(str(err))
    st.stop()

pf, m, panel = res.portfolio, res.metrics, res.panel
st.title("Promo analytics")
st.caption(
    "Demo data"
    if demo
    else f"Customer {cid} · {info['promo_rows']:,} promotion rows, "
    f"{info['secondary_rows']:,} with a secondary (retail) window · testing the {promo_window} window"
)
if promo_window == "secondary" and not demo and info["secondary_rows"] == 0:
    st.warning(
        "This customer has no secondary promo windows. Switch to the primary window."
    )

tab_overview, tab_dep, tab_shift, tab_series = st.tabs(
    ["Overview", "Promo dependency", "Promo shift", "Series explorer"]
)

# --------------------------------------------------------------------- overview

with tab_overview:
    c = st.columns(5)
    c[0].metric(
        "Series",
        f"{pf['series']:,}",
        f"{pf['promoted_series']:,} with promos",
        delta_color="off",
    )
    c[1].metric(
        "Sold during promos",
        f"{pf['promo_volume_share']:.0%}",
        f"{pf['promo_volume_share_adj'] - pf['promo_volume_share']:+.0%} after shift correction",
        delta_color="off",
    )
    c[2].metric(
        "Incremental volume",
        f"{pf['incremental_share']:.0%}",
        help="Volume above baseline during (shift-adjusted) promos, as a share of all volume.",
    )
    c[3].metric(
        "Volume in high-dependency series", f"{pf['high_dependency_volume']:.0%}"
    )
    c[4].metric(
        "Shifted volume",
        f"{pf['shifted_volume_share']:.0%}",
        help=f"Share of volume in series whose sales peak before or after recorded promos "
        f"({pf['tested_series']} series had enough promos to test).",
    )

    notes = []
    inc = pf["incremental_share"]
    level = "strongly" if inc >= high else "moderately" if inc >= low else "barely"
    notes.append(
        f"The portfolio is **{level} promo-driven**: {inc:.0%} of volume is promo uplift."
    )
    shifted = m[m["best_lag"].fillna(0) != 0]
    if len(shifted):
        typical = shifted["best_lag"].mode().iloc[0]
        notes.append(
            f"**{len(shifted)} series** peak away from the recorded promo dates, most often "
            f"**{describe_lag(typical, unit)}**. Sales before a promo usually mean sell-in "
            "(a distributor or DC stocking up for a store promotion); sales after usually mean the "
            "promo dates are announcement dates or the sales data lags. Shift promo features by "
            "that lag for these series before training."
        )
        if shifted["best_lag"].abs().eq(max_lag).any():
            notes.append(
                f"⚠️ Some shifts sit at the edge of the tested range (±{max_lag} {unit}s). "
                "Increase *Largest shift to test* under Advanced."
            )
    else:
        notes.append("Promo effects line up with the recorded promo dates.")
    if promo_window == "secondary":
        notes.append(
            "You are testing against the retail shelf period. For a producer, orders "
            "*before* it are expected; compare the shift with the gap between the primary "
            "and secondary windows to check the stored ordering window is right."
        )
    st.markdown("\n\n".join(notes))

    st.subheader("Which shift explains the uplift best?")
    lag_df = res.pooled_lag_scores.reset_index()
    st.altair_chart(
        alt.Chart(lag_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "lag:O",
                title=f"Sales timing vs. recorded promo ({unit}s, negative = before)",
            ),
            y=alt.Y("score:Q", title="Uplift during promo vs. outside"),
            color=alt.condition(
                alt.datum.lag == pf["shift"], alt.value("#c0392b"), alt.value("#7f8c8d")
            ),
            tooltip=["lag", alt.Tooltip("score:Q", format=".2f")],
        ),
        width="stretch",
    )
    st.caption(
        "Portfolio-wide. A mix of aligned and shifted series can put the peak at 0 — "
        "see the per-series distribution in Promo shift."
    )

# ------------------------------------------------------------------- dependency

with tab_dep:
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
                    range=["#c0392b", "#e67e22", "#2980b9", "#bdc3c7"],
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
            "series": "Series",
            "dependency": "Dependency",
            "volume": st.column_config.NumberColumn("Volume", format="%,.0f"),
            "promo_events": "Promos",
            "promo_period_share": st.column_config.ProgressColumn(
                "Time on promo", format="percent", min_value=0, max_value=1
            ),
            "promo_volume_share": st.column_config.ProgressColumn(
                "Sold on promo", format="percent", min_value=0, max_value=1
            ),
            "promo_volume_share_adj": st.column_config.ProgressColumn(
                "Sold on promo (shift-adj.)", format="percent", min_value=0, max_value=1
            ),
            "incremental_share": st.column_config.ProgressColumn(
                "Incremental", format="percent", min_value=0, max_value=1
            ),
            "promo_lift": st.column_config.NumberColumn("Lift ×", format="%.2f"),
            "best_lag": st.column_config.NumberColumn(f"Shift ({unit}s)", format="%+d"),
        },
    )
    st.download_button(
        "Download table as CSV",
        m.to_csv(index=False).encode(),
        file_name=f"promo_metrics_{cid or 'demo'}.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------------ shift

with tab_shift:
    left, right = st.columns(2)
    with left:
        st.subheader("Average sales around a promo start")
        prof = res.event_profile
        if prof.empty:
            st.info("No isolated promo starts to average.")
        else:
            x = alt.X("offset:Q", title=f"{unit.title()}s from recorded promo start")
            st.altair_chart(
                alt.Chart(prof)
                .mark_area(opacity=0.2, color="#2980b9")
                .encode(x=x, y="lo:Q", y2="hi:Q")
                + alt.Chart(prof)
                .mark_line(point=True, color="#2980b9")
                .encode(
                    x=x,
                    y=alt.Y("mean:Q", title="Uplift (× average sales)"),
                    tooltip=["offset", alt.Tooltip("mean:Q", format=".2f"), "events"],
                )
                + alt.Chart(pd.DataFrame({"x": [0]}))
                .mark_rule(strokeDash=[4, 4])
                .encode(x="x:Q"),
                width="stretch",
            )
            st.caption(
                f"Based on {int(prof['events'].max())} promos with no other promo just before. "
                "Uplift before 0 means sales anticipate the promo; a dip after the promo means "
                "pull-forward or pantry loading."
            )
    with right:
        st.subheader("Detected shift per series")
        dist = (
            m["best_lag"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .rename_axis("lag")
            .reset_index(name="series")
        )
        if dist.empty:
            st.info(f"No series has {min_events}+ promos yet.")
        else:
            st.altair_chart(
                alt.Chart(dist)
                .mark_bar(color="#7f8c8d")
                .encode(
                    x=alt.X(
                        "lag:O", title=f"Shift ({unit}s, negative = sales before promo)"
                    ),
                    y=alt.Y("series:Q", title="Series"),
                    tooltip=["lag", "series"],
                ),
                width="stretch",
            )
            if by_location:
                by_loc = (
                    m.dropna(subset=["best_lag"])
                    .groupby("location")
                    .agg(
                        series=("series", "size"),
                        median_shift=("best_lag", "median"),
                        volume=("volume", "sum"),
                    )
                    .sort_values("volume", ascending=False)
                )
                st.dataframe(by_loc, width="stretch")
                st.caption(
                    "Accounts with a consistent negative shift are likely distributors or DCs."
                )

# --------------------------------------------------------------- series explorer

with tab_series:
    pick = st.selectbox("Series (sorted by volume)", m["series"])
    row = m.set_index("series").loc[pick]
    d = panel[panel["series"] == pick].assign(
        period_end=lambda x: x["period"] + res.settings.period_length
    )
    c = st.columns(4)
    c[0].metric("Promos", int(row["promo_events"]))
    c[1].metric(
        "Incremental",
        (
            f"{row['incremental_share']:.0%}"
            if pd.notna(row["incremental_share"])
            else "–"
        ),
    )
    c[2].metric(
        "Lift", f"{row['promo_lift']:.2f}×" if pd.notna(row["promo_lift"]) else "–"
    )
    c[3].metric("Shift", describe_lag(row["best_lag"], unit))

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
                scale=alt.Scale(
                    domain=["qty", "baseline"], range=["#2c3e50", "#95a5a6"]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["period:T", alt.Tooltip("value:Q", format=",.0f"), "measure:N"],
        )
    )
    st.altair_chart(
        (bands + lines).properties(height=380).interactive(bind_y=False),
        width="stretch",
    )
    st.caption("Orange bands are recorded promo periods.")

    one = (
        res.lag_scores.loc[tuple(row[k] for k in res.settings.keys)]
        if by_location
        else res.lag_scores.loc[row["product"]]
    )
    st.altair_chart(
        alt.Chart(one.rename_axis("lag").reset_index(name="score"))
        .mark_bar(color="#7f8c8d")
        .encode(
            x=alt.X("lag:O", title=f"Shift ({unit}s)"),
            y=alt.Y("score:Q", title="Uplift score"),
        ),
        width="stretch",
    )
