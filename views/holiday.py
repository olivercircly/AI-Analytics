"""Holidays: sales around each public holiday vs. a normal week."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import calendar_effects as ce

from views.common import INK, Context, pct
from views.data import holiday_result


def render(ctx: Context) -> None:
    S, H, src, country = ctx.S, ctx.H, ctx.src, ctx.country
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
