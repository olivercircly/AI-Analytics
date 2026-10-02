"""Overview: headline metrics and one finding per analysis."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import calendar_effects as ce
import seasonality as se

from views.common import INK, Context, pct, plural
from views.data import panel


def render(ctx: Context) -> None:
    S, B, P, Q, H = ctx.S, ctx.B, ctx.P, ctx.Q, ctx.H
    src, unit, horizon = ctx.src, ctx.unit, ctx.horizon
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

    qs, lat = Q["summary"], Q["latency"]
    fc = qs["flag_counts"]
    dq_bits = []
    if qs["days_old"] > 2:
        dq_bits.append(f"newest sales are {plural(qs['days_old'], 'day')} old")
    if lat and lat["k95"] is not None and not lat["bulk_loaded"]:
        dq_bits.append(f"the last {plural(lat['k95'], 'day')} are usually incomplete")
    if fc.get("missing", 0) + fc.get("partial", 0):
        dq_bits.append(
            f"{plural(fc.get('missing', 0) + fc.get('partial', 0), 'account-day')} missing or partly loaded"
        )
    if len(qs["silent_accounts"]):
        dq_bits.append(
            f"{plural(len(qs['silent_accounts']), 'account')} stopped sending data"
        )
    findings = [
        (
            ("**Data quality:** " + "; ".join(dq_bits) + " (see Data quality).")
            if dq_bits
            else "**Data quality:** no gaps or delays found."
        )
    ]
    findings.append(
        f"**Concentration:** {n80:,} of {len(vol):,} articles make up 80% of volume."
    )
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
    shifted = ctx.shifted
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
