"""Promotions: dependency, incremental volume and the promo shift."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from views.common import GREY, ORANGE, BLUE, RED, WINDOWS, Context, pct


def render(ctx: Context) -> None:
    P, unit, cid, info, demo = ctx.P, ctx.unit, ctx.cid, ctx.info, ctx.demo
    promo_window, by_location, max_lag = ctx.promo_window, ctx.by_location, ctx.max_lag
    min_events, low, high = ctx.min_events, ctx.low, ctx.high
    pp, shifted = P.portfolio, ctx.shifted
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
