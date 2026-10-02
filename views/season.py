"""Seasonality: weekday and yearly patterns, portfolio and per article."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import seasonality as se

from views.common import BLUE, INK, ORANGE, Context, pct


def render(ctx: Context) -> None:
    S = ctx.S
    sp, wp = S["portfolio"], S["weekday_profile"]
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
