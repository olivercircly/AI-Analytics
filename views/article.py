"""Article mode: how one article developed, across all its accounts, and its anomalies."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import anomalies as ab
from views.common import (
    BLUE,
    GREY,
    INK,
    ORANGE,
    RED,
    ArticleContext,
    pct,
    plural,
    virtual_note,
)

KIND_COLORS = {
    "spike": RED,
    "drop": ORANGE,
    "gap": "#8e44ad",
    "stopped": "#8e44ad",
    "level shift": BLUE,
    "flat promo": GREY,
}


def _when(d: pd.Timestamp, unit: str) -> str:
    return f"week of {d:%d %b %Y}" if unit == "week" else f"{d:%a %d %b %Y}"


def _findings(ctx: ArticleContext) -> list[str]:
    R, unit = ctx.R, ctx.unit
    ev = R["events"]
    out = []
    for _, e in ev[ev["kind"] == "stopped"].iterrows():
        who = "The article" if e["account"] == "all" else f"Account **{e['account']}**"
        out.append(
            f"{who} has had **no sales since {e['start'] - pd.Timedelta(days=1):%d %b %Y}**, "
            f"about {e['expected']:,.0f} units below its usual rate. Delisted, out of stock, or a broken feed?"
        )
    for _, e in ev[ev["kind"] == "level shift"].iterrows():
        acc = R["level_shifts"].set_index("start").loc[e["start"]]
        who = (
            f", mostly account **{acc['account']}**"
            if pd.notna(acc["account"]) and len(R["accounts"]) > 2
            else ""
        )
        out.append(
            f"**Level shift** in the week of {e['start']:%d %b %Y}: {pct(e['change'], True)} "
            f"({e['expected']:,.1f} → {e['qty']:,.1f} units per week, seasonality removed){who}."
        )
    gaps = ev[ev["kind"] == "gap"]
    if len(gaps):
        longest = gaps.assign(len=gaps["end"] - gaps["start"]).sort_values("len").iloc[-1]
        out.append(
            f"**{plural(len(gaps), 'gap')}** with no sales where some were expected; the longest runs "
            f"from {longest['start']:%d %b} to {longest['end']:%d %b %Y} (about {longest['expected']:,.0f} units missing). "
            "Stock-outs or missing data: mask them when training."
        )
    for kind, word in [("spike", "spike"), ("drop", "drop")]:
        k = ev[ev["kind"] == kind]
        if len(k):
            top = k.loc[(k["qty"] - k["expected"]).abs().idxmax()]
            out.append(
                f"**{plural(len(k), word)}** outside promotions; the biggest is {_when(top['start'], unit)} "
                f"({top['qty']:,.0f} units vs. {top['expected']:,.0f} expected). Check for data-entry errors or one-off orders."
            )
    flat = ev[ev["kind"] == "flat promo"]
    if len(flat):
        out.append(
            f"**{plural(len(flat), 'promo')} without visible uplift** (see Promotions below)."
        )
    return out


def _development_chart(ctx: ArticleContext) -> alt.Chart:
    R, unit = ctx.R, ctx.unit
    step = pd.Timedelta(days=1 if unit == "day" else 7)
    d = R["panel"].assign(period_end=lambda x: x["period"] + step)
    ev = R["events"].assign(
        end_excl=lambda x: x["end"] + step,
        what=lambda x: x["kind"].map(ab.KINDS),
    )
    x = alt.X("period:T", title=None)
    promo = (
        alt.Chart(d[d["promo"]])
        .mark_rect(color="#f39c12", opacity=0.2)
        .encode(x=x, x2="period_end:T")
    )
    lines = (
        alt.Chart(d)
        .transform_fold(["qty", "expected"], as_=["measure", "value"])
        .mark_line()
        .encode(
            x=x,
            y=alt.Y("value:Q", title=f"Units per {unit}"),
            color=alt.Color(
                "measure:N",
                scale=alt.Scale(domain=["qty", "expected"], range=[INK, GREY]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeDash=alt.condition(
                alt.datum.measure == "expected", alt.value([4, 3]), alt.value([1, 0])
            ),
            tooltip=[
                alt.Tooltip("period:T", title=unit.title()),
                "measure:N",
                alt.Tooltip("value:Q", format=",.1f"),
            ],
        )
    )
    color = alt.Color(
        "kind:N",
        title=None,
        scale=alt.Scale(domain=list(KIND_COLORS), range=list(KIND_COLORS.values())),
        legend=alt.Legend(orient="top"),
    )
    tip = [
        "kind:N",
        "what:N",
        alt.Tooltip("start:T", title="from"),
        alt.Tooltip("end:T", title="to"),
        alt.Tooltip("qty:Q", format=",.1f"),
        alt.Tooltip("expected:Q", format=",.1f"),
        "account:N",
    ]
    spans = (
        alt.Chart(ev[ev["kind"].isin(["gap", "stopped", "flat promo"])])
        .mark_rect(opacity=0.25)
        .encode(x="start:T", x2="end_excl:T", color=color, tooltip=tip)
    )
    rules = (
        alt.Chart(ev[ev["kind"] == "level shift"])
        .mark_rule(strokeWidth=2, strokeDash=[6, 3])
        .encode(x="start:T", color=color, tooltip=tip)
    )
    points = (
        alt.Chart(ev[ev["kind"].isin(["spike", "drop"])])
        .mark_point(size=120, filled=True)
        .encode(x="start:T", y="qty:Q", color=color, tooltip=tip)
    )
    return (
        (promo + spans + lines + rules + points)
        .resolve_scale(color="independent")
        .properties(height=360)
        .interactive(bind_y=False)
    )


def render(ctx: ArticleContext) -> None:
    R, sales, unit, info = ctx.R, ctx.sales, ctx.unit, ctx.info
    acc, ev, weekly = R["accounts"], R["events"], R["weekly"]
    number = sales["article_number"].iloc[0]
    st.title(f"Article {number}")
    st.caption(
        ("Demo data. " if ctx.demo else "")
        + f"articleId {ctx.article_id} · customer {info['cid']} · "
        f"{plural(int((acc['location'] != 'all').sum()), 'account')} · sales from "
        f"{sales['date'].min():%d %b %Y} to {sales['date'].max():%d %b %Y} · "
        f"{plural(info['promo_rows'], 'promotion row')}"
        + (
            f" ({ctx.promo_window} window)"
            if info["promo_rows"]
            else ""
        )
        + "."
        + virtual_note(info.get("virtual"))
    )
    if not ctx.demo and info["cid"]:
        virtual = "&virtual=1" if info.get("virtual") is not None else ""
        st.markdown(f"[Open the customer view for {info['cid']}](?cid={info['cid']}{virtual})")

    wt = weekly.set_index("period")["qty"]
    last52 = wt.iloc[-52:].sum()
    yoy = last52 / wt.iloc[-104:-52].sum() - 1 if len(wt) >= 104 else np.nan
    article = acc.set_index("location").loc["all"]
    m, shift = R["metrics"], R["shift"]
    c = st.columns(6)
    c[0].metric(
        "Volume, last 52 weeks",
        f"{last52:,.0f}",
        pct(yoy, True) + " vs. year before" if pd.notna(yoy) else None,
    )
    c[1].metric(
        "Last sale",
        f"{article['last_sale']:%d %b %Y}",
        f"{plural(int(article['days_silent']), 'day')} ago",
        delta_color="off" if article["days_silent"] <= 2 else "inverse",
    )
    stopped = (acc["location"] != "all") & acc["status"].str.startswith("stopped")
    c[2].metric(
        "Accounts",
        f"{int((acc['location'] != 'all').sum()):,}",
        f"{int(stopped.sum()):,} stopped" if stopped.any() else None,
        delta_color="inverse" if stopped.any() else "off",
    )
    c[3].metric(
        "Promo uplift",
        pct(m["incremental_share"]),
        f"{int(m['promo_events']):,} promos",
        delta_color="off",
    )
    trusted = pd.notna(m["best_lag"])
    c[4].metric(
        "Promo shift",
        "aligned"
        if shift == 0
        else f"{abs(shift)} {unit}{'s' if abs(shift) > 1 else ''} {'before' if shift < 0 else 'after'}",
        None if trusted else "too few promos to test",
        delta_color="off",
        help="Sales timing vs. recorded promo dates. With fewer promos than set under "
        "Promotion settings, the recorded dates are assumed to be right.",
    )
    c[5].metric("Anomalies", f"{len(ev):,}")

    notes = _findings(ctx)
    st.markdown("\n".join(f"- {n}" for n in notes) if notes else "No anomalies found.")

    st.subheader("Development")
    st.altair_chart(_development_chart(ctx), width="stretch")
    st.caption(
        "Expected = the article's normal level (rolling mean outside promos and long gaps"
        + (", times its weekday pattern" if unit == "day" else "")
        + "). Orange bands are recorded promos; shaded spans, dashed lines and dots mark anomalies."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Year over year")
        overlay = weekly.assign(
            year=weekly["period"].dt.isocalendar().year.astype(int),
            week=weekly["period"].dt.isocalendar().week.astype(int),
        )
        st.altair_chart(
            alt.Chart(overlay)
            .mark_line()
            .encode(
                x=alt.X("week:Q", title="Week of year", scale=alt.Scale(domain=[1, 53])),
                y=alt.Y("qty:Q", title="Units per week"),
                color=alt.Color("year:O", title=None),
                tooltip=["year", "week", alt.Tooltip("qty:Q", format=",.0f")],
            ),
            width="stretch",
        )
    with right:
        st.subheader("By account")
        ba = R["by_account"]
        top = acc.loc[acc["location"] != "all", "location"].head(8).tolist()
        ba = ba.assign(account=ba["location"].where(ba["location"].isin(top), "other"))
        ba = ba.groupby(["period", "account"], as_index=False)["qty"].sum()
        st.altair_chart(
            alt.Chart(ba)
            .mark_area()
            .encode(
                x=alt.X("period:T", title=None),
                y=alt.Y("qty:Q", title="Units per week", stack=True),
                color=alt.Color(
                    "account:N", title=None, sort=top + ["other"],
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("period:T", title="Week of"),
                    "account",
                    alt.Tooltip("qty:Q", format=",.0f"),
                ],
            ),
            width="stretch",
        )

    st.subheader("Anomalies")
    if ev.empty:
        st.info("None found.")
    else:
        st.dataframe(
            ev.assign(what=ev["kind"].map(ab.KINDS)),
            hide_index=True,
            width="stretch",
            column_config={
                "kind": "Type",
                "start": st.column_config.DateColumn("From"),
                "end": st.column_config.DateColumn("To"),
                "account": "Account",
                "qty": st.column_config.NumberColumn("Units", format="%,.1f"),
                "expected": st.column_config.NumberColumn("Expected", format="%,.1f"),
                "change": st.column_config.NumberColumn("vs. expected", format="percent"),
                "what": "Meaning",
            },
        )
        st.caption(
            "Level shifts: Units / Expected are the weekly level after / before. "
            "Gaps and stopped: Expected is the total that should have sold."
        )
        st.download_button(
            "Download anomalies as CSV",
            ev.to_csv(index=False).encode(),
            file_name=f"anomalies_{ctx.article_id}.csv",
            mime="text/csv",
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Promotions")
        pr = R["promos"]
        if pr.empty:
            st.info("No promotions recorded for this article.")
        else:
            st.dataframe(
                pr.assign(flat=pr["flat"].map({True: "no uplift", False: ""})),
                hide_index=True,
                width="stretch",
                column_config={
                    "promo_start": st.column_config.DateColumn("From"),
                    "promo_end": st.column_config.DateColumn("To"),
                    "location": "Account",
                    "qty": st.column_config.NumberColumn("Units", format="%,.0f"),
                    "baseline": st.column_config.NumberColumn("Expected", format="%,.0f"),
                    "lift": st.column_config.NumberColumn("Lift ×", format="%.2f"),
                    "flat": "",
                },
            )
            st.caption(
                f"Lift per promo day, measured {plural(abs(shift), unit)} "
                f"{'earlier' if shift < 0 else 'later'} than recorded."
                if shift
                else "Lift per promo day on the recorded dates."
            )
    with right:
        st.subheader("Accounts")
        st.dataframe(
            acc.drop(columns=["missed"]),
            hide_index=True,
            width="stretch",
            column_config={
                "location": "Account",
                "volume": st.column_config.NumberColumn("Volume", format="%,.0f"),
                "first_sale": st.column_config.DateColumn("First sale"),
                "last_sale": st.column_config.DateColumn("Last sale"),
                "days_silent": "Days without sales",
                "status": "Status",
                "volume_share": st.column_config.ProgressColumn(
                    "Share", format="percent", min_value=0, max_value=1
                ),
            },
        )
