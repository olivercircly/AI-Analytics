"""Article explorer: one article's sales, baseline, promos and seasonal pattern."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import seasonality as se

from views.common import BLUE, GREY, INK, ORANGE, Context, pct


def render(ctx: Context) -> None:
    S, B, P, unit = ctx.S, ctx.B, ctx.P, ctx.unit
    m = P.metrics
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
