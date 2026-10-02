"""Virtual accounts: do they sell disjoint sets of articles?"""

from __future__ import annotations

import altair as alt
import streamlit as st

import analysis as an
from views.common import BLUE, INK, Context, pct, plural
from views.data import load

MAX_HEATMAP = 30  # accounts; beyond that the cells get too small to read


def render(ctx: Context) -> None:
    O = ctx.O
    acc, art, pairs = O["accounts"], O["articles"], O["pairs"]
    sales, _, _ = load(ctx.src)
    n_va = len(acc)
    if not n_va:
        st.info("No virtual account has sales here.")
        return

    inside = art[art["groups"] > 0]
    shared = art[art["groups"] > 1]
    both = inside[inside["outside"]]
    va_volume = acc["volume"].sum()
    shared_volume = (acc["volume"] * acc["shared_volume_share"]).sum()

    c = st.columns(4)
    c[0].metric("Virtual accounts", f"{n_va:,}")
    c[1].metric(
        "Articles in more than one",
        f"{len(shared):,}",
        f"of {len(inside):,} sold in virtual accounts",
        delta_color="off",
    )
    c[2].metric(
        "Volume in shared articles",
        pct(shared_volume / va_volume) if va_volume else "–",
        help="Share of virtual-account volume from articles sold in more than one virtual account.",
    )
    c[3].metric(
        "Also sold outside",
        f"{len(both):,}",
        "articles at stores in no virtual account",
        delta_color="off",
    )

    if n_va == 1:
        st.info(
            f"Only one virtual account ({acc['location'].iloc[0]}) has sales here, so there "
            "is no other virtual account to compare it with."
        )
    elif shared.empty:
        st.success("**Disjoint**: every article sells in exactly one virtual account.")
    else:
        st.warning(
            f"**Not disjoint**: {plural(len(shared), 'article')} "
            f"({pct(shared_volume / va_volume)} of virtual-account volume) sell in more than "
            f"one virtual account, up to {plural(int(shared['groups'].max()), 'virtual account')} "
            "for one article. Listed below."
        )
    if len(both):
        st.markdown(
            f"{plural(len(both), 'article')} sold in a virtual account also sell at stores "
            "that belong to no virtual account."
        )
    st.caption(
        f"An article counts for an account if it sold there at least once between "
        f"{sales['date'].min():%d %b %Y} and {sales['date'].max():%d %b %Y}. "
        "Move the start date to look at a more recent assortment."
    )

    order = acc["location"].tolist() + (
        [an.OUTSIDE] if (pairs["a"] == an.OUTSIDE).any() else []
    )
    off = pairs[(pairs["a"] != pairs["b"]) & pairs["a"].isin(order[:MAX_HEATMAP])
                & pairs["b"].isin(order[:MAX_HEATMAP])]
    if len(off):  # nothing to draw when no two accounts share an article
        st.subheader("Articles sold in both")
        shown = order[:MAX_HEATMAP]
        own = pairs[pairs["a"] == pairs["b"]].set_index("a")["shared"]
        cells = off.assign(articles_a=off["a"].map(own), articles_b=off["b"].map(own))
        tilt = len(shown) > 6  # horizontal labels while they fit
        band = alt.Scale(domain=shown, paddingInner=0.06)
        base = alt.Chart(cells).encode(
            x=alt.X("b:N", title=None, scale=band,
                    axis=alt.Axis(labelAngle=-40 if tilt else 0, labelLimit=180, orient="top")),
            y=alt.Y("a:N", title=None, scale=band, axis=alt.Axis(labelLimit=220)),
        )
        labelled = len(shown) <= 12  # numbers in the cells while they still fit
        heat = base.mark_rect(cornerRadius=2).encode(
            color=alt.Color(
                "shared:Q",
                title="Shared articles",
                legend=None if labelled else alt.Legend(),
                scale=alt.Scale(range=["#a9cce3", BLUE], domainMin=0),
            ),
            tooltip=[
                alt.Tooltip("a:N", title="Account"),
                alt.Tooltip("b:N", title="and"),
                alt.Tooltip("shared:Q", title="Shared articles", format=",.0f"),
                alt.Tooltip("articles_a:Q", title="Articles in first", format=",.0f"),
                alt.Tooltip("articles_b:Q", title="Articles in second", format=",.0f"),
                alt.Tooltip("jaccard:Q", title="Overlap (Jaccard)", format=".0%"),
            ],
        )
        layers = [heat]
        if labelled:
            layers.append(
                base.mark_text(fontSize=11).encode(
                    text=alt.Text("shared:Q", format=",.0f"),
                    color=alt.condition(
                        f"datum.shared > {cells['shared'].max() / 2}",
                        alt.value("white"),
                        alt.value(INK),
                    ),
                )
            )
        # the chart is sized with the axes included, so leave room for the labels
        st.altair_chart(
            alt.layer(*layers).properties(height=36 * len(shown) + (130 if tilt else 40)),
            width="stretch",
        )
        st.caption(
            "Each cell counts the articles sold in both accounts; empty cells share none. "
            + (f"Top {MAX_HEATMAP} accounts by volume. " if len(order) > MAX_HEATMAP else "")
            + "Hover for each account's article count and the Jaccard overlap "
            "(shared ÷ articles in either)."
        )

    st.subheader("Per virtual account")
    st.dataframe(
        acc,
        hide_index=True,
        width="stretch",
        column_config={
            "location": "Virtual account",
            "articles": "Articles",
            "exclusive": st.column_config.NumberColumn(
                "Only here", help="Sold in no other virtual account"
            ),
            "shared": st.column_config.NumberColumn(
                "Shared", help="Also sold in another virtual account"
            ),
            "volume": st.column_config.NumberColumn("Volume", format="%,.0f"),
            "shared_volume_share": st.column_config.ProgressColumn(
                "Volume in shared articles", format="percent", min_value=0, max_value=1
            ),
        },
    )

    articles_config = {
        "product": "Article",
        "groups": "Virtual accounts",
        "where": "Sold in",
        "outside": st.column_config.CheckboxColumn(
            "Also outside", help="Also sold at stores in no virtual account"
        ),
        "volume": st.column_config.NumberColumn("Volume", format="%,.0f"),
    }
    if len(shared):
        st.subheader("Articles in more than one virtual account")
        st.dataframe(shared, hide_index=True, width="stretch", column_config=articles_config)
    st.download_button(
        "Download all articles as CSV",
        art.to_csv(index=False).encode(),
        file_name=f"virtual_account_articles_{ctx.cid or 'demo'}.csv",
        mime="text/csv",
    )
