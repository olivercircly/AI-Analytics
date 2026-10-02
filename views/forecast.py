"""Forecastability: backtest of simple benchmark methods."""

from __future__ import annotations

import altair as alt
import streamlit as st

from views.common import BLUE, GREY, ORANGE, RED, Context, pct


def render(ctx: Context) -> None:
    B, unit, freq = ctx.B, ctx.unit, ctx.freq
    horizon, test_periods = ctx.horizon, ctx.test_periods
    default_h, default_test = ctx.default_h, ctx.default_test
    bp = B["portfolio"]
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
