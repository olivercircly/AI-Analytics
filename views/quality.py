"""Data quality: freshness, arrival delays, missing days, accounts, rewrites."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import data_quality as dq

from views.common import BLUE, GREY, ORANGE, RED, Context, pct, plural


def render(ctx: Context) -> None:
    Q, cid = ctx.Q, ctx.cid
    qs, lat, ch, acc = Q["summary"], Q["latency"], Q["changes"], Q["accounts"]
    fc = qs["flag_counts"]
    c = st.columns(6)
    c[0].metric(
        "Newest sales day",
        f"{qs['last_day']:%d %b %Y}",
        f"{plural(qs['days_old'], 'day')} ago",
        delta_color="off" if qs["days_old"] <= 2 else "inverse",
    )
    c[1].metric(
        "Complete after",
        (
            "–"
            if not lat or lat["k95"] is None or lat["bulk_loaded"]
            else plural(lat["k95"], "day")
        ),
        help="Days after a sales day until 95% of its rows have arrived (measured on the last 90 mature days).",
    )
    c[2].metric(
        "Missing days",
        f"{fc.get('missing', 0):,}",
        f"{fc.get('partial', 0):,} partly loaded",
        delta_color="off",
    )
    c[3].metric(
        "Silent accounts",
        f"{len(qs['silent_accounts']):,}",
        f"of {len(acc):,}",
        delta_color="off",
    )
    c[4].metric("Rows changed after loading", pct(ch["updated"]) if ch else "–")
    c[5].metric("Rows deleted", pct(ch["deleted"]) if ch else "–")

    todo = []
    if qs["days_old"] > 2:
        todo.append(
            f"The newest sales day is **{plural(qs['days_old'], 'day')} old**. Check the data feed before forecasting."
        )
    if lat and lat["bulk_loaded"]:
        todo.append(
            "Recent history was loaded in bulk (re-import or backfill), so arrival delays can't be measured."
        )
    elif lat and lat["k95"]:
        todo.append(
            f"Data trickles in: {pct(lat['curve'].loc[1, 'share'])} of a day's rows are there the next day, "
            f"95% after **{plural(lat['k95'], 'day')}**. Treat the last {plural(lat['k95'], 'day')} as incomplete when training, "
            "or they look like a demand drop and pull the forecast down."
        )
    if fc.get("missing", 0) + fc.get("partial", 0):
        todo.append(
            f"**{plural(fc.get('missing', 0) + fc.get('partial', 0), 'account-day')}** missing or partly loaded "
            f"(about {pct(qs['missing_volume_share'])} of volume). Mask them as unknown instead of "
            "training on them as zero demand. Download the list below."
        )
    if len(Q["outages"]):
        todo.append(
            f"**System-wide outages** on {', '.join(f'{d:%d %b %Y}' for d in Q['outages']['date'])}: "
            "most accounts have no data at all."
        )
    for _, r in qs["silent_accounts"].iterrows():
        todo.append(
            f"Account **{r['location']}** ({pct(r['volume_share'])} of volume) has had no sales since "
            f"{r['last_sale']:%d %b %Y}. Closed, or the feed broke?"
        )
    if ch and ch["updated"] > 0.01:
        todo.append(
            f"{pct(ch['updated'])} of rows were changed after loading. History is rewritten, so backtests on "
            "today's data can look better than the forecasts did at the time."
        )
    if len(Q["outliers"]):
        todo.append(
            f"{plural(int((Q['outliers']['kind'] == 'spike').sum()), 'article-day')} over 8× the article's usual "
            f"level outside promotions, {int((Q['outliers']['kind'] == 'net negative').sum()):,} net negative. "
            "Check them for data-entry errors."
        )
    st.markdown("\n".join(f"- {t}" for t in todo) if todo else "No problems found.")

    left, right = st.columns(2)
    with left:
        st.subheader("When does a day's data arrive?")
        if lat and not lat["curve"].empty and not lat["bulk_loaded"]:
            curve = lat["curve"][lat["curve"]["day"] <= 14]
            st.altair_chart(
                alt.Chart(curve)
                .mark_line(point=True, color=BLUE)
                .encode(
                    x=alt.X("day:Q", title="Days after the sales day"),
                    y=alt.Y(
                        "share:Q",
                        title="Share of rows arrived",
                        axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    tooltip=["day", alt.Tooltip("share:Q", format=".1%")],
                )
                + alt.Chart(pd.DataFrame({"y": [0.95]}))
                .mark_rule(strokeDash=[4, 4], color=GREY)
                .encode(y="y:Q"),
                width="stretch",
            )
        else:
            st.info("No arrival data for recent days.")
    with right:
        st.subheader("Rows loaded per week")
        if Q["arrivals"] is not None and len(Q["arrivals"]):
            arr = Q["arrivals"].assign(
                week=lambda x: x["arrival"]
                - pd.to_timedelta(x["arrival"].dt.dayofweek, unit="D")
            )
            arr = arr.groupby(["week", "kind"], as_index=False)["n_rows"].sum()
            st.altair_chart(
                alt.Chart(arr)
                .mark_bar()
                .encode(
                    x=alt.X("week:T", title="Loaded in week"),
                    y=alt.Y("n_rows:Q", title="Rows", stack=True),
                    color=alt.Color(
                        "kind:N",
                        title=None,
                        scale=alt.Scale(
                            domain=["regular", "backfill / re-import"],
                            range=[BLUE, ORANGE],
                        ),
                        legend=alt.Legend(orient="top"),
                    ),
                    tooltip=[
                        alt.Tooltip("week:T", title="Week of"),
                        "kind",
                        alt.Tooltip("n_rows:Q", format=","),
                    ],
                ),
                width="stretch",
            )
            st.caption(
                "Backfill = rows loaded more than 30 days after their sales day."
            )

    st.subheader("Flagged days by account")
    flagged = Q["flagged"]
    if flagged.empty:
        st.info("No missing, partial or unusual days.")
    else:
        top_acc = acc["location"].head(30).tolist()
        shown = flagged[flagged["location"].isin(top_acc)]
        st.altair_chart(
            alt.Chart(shown)
            .mark_tick(thickness=3, size=14)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("location:N", title="Account", sort=top_acc),
                color=alt.Color(
                    "flag:N",
                    title=None,
                    scale=alt.Scale(
                        domain=list(dq.FLAGS), range=[RED, ORANGE, "#f1c40f", BLUE]
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=[
                    "location",
                    alt.Tooltip("date:T", format="%a %d %b %Y"),
                    "flag",
                    alt.Tooltip("qty:Q", format=",.0f"),
                    alt.Tooltip("usual_qty:Q", title="usual", format=",.0f"),
                ],
            )
            .properties(height=max(120, 22 * len(set(shown["location"])))),
            width="stretch",
        )
        st.caption(
            " ".join(f"**{k}**: {v}." for k, v in dq.FLAGS.items())
            + " Only days the account normally trades on, excluding holidays. Top 30 accounts by volume."
        )
        st.download_button(
            "Download flagged days as CSV",
            flagged.to_csv(index=False).encode(),
            file_name=f"flagged_days_{cid or 'demo'}.csv",
            mime="text/csv",
        )

    st.subheader("Accounts")
    st.dataframe(
        acc,
        hide_index=True,
        width="stretch",
        column_config={
            "location": "Account",
            "first_day": st.column_config.DateColumn("First day"),
            "last_sale": st.column_config.DateColumn("Last sale"),
            "trading_days": "Trading days",
            "volume": st.column_config.NumberColumn("Volume", format="%,.0f"),
            "volume_share": st.column_config.ProgressColumn(
                "Share of volume", format="percent", min_value=0, max_value=1
            ),
            "missing": "Missing",
            "partial": "Partial",
            "low": "Low",
            "spike": "Spike",
            "days_silent": "Days without sales",
            "status": "Status",
        },
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Changes after loading, by sales month")
        if ch and len(ch["monthly"]):
            mon = ch["monthly"].melt(
                "month",
                ["updated_share", "deleted_share", "negative_share"],
                "what",
                "share",
            )
            mon["what"] = mon["what"].map(
                {
                    "updated_share": "changed",
                    "deleted_share": "deleted",
                    "negative_share": "negative quantity",
                }
            )
            st.altair_chart(
                alt.Chart(mon)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month:T", title=None),
                    y=alt.Y(
                        "share:Q", title="Share of rows", axis=alt.Axis(format="%")
                    ),
                    color=alt.Color(
                        "what:N",
                        title=None,
                        scale=alt.Scale(range=[BLUE, RED, GREY]),
                        legend=alt.Legend(orient="top"),
                    ),
                    tooltip=[
                        alt.Tooltip("month:T", format="%b %Y"),
                        "what",
                        alt.Tooltip("share:Q", format=".2%"),
                    ],
                ),
                width="stretch",
            )
    with right:
        st.subheader("Suspicious article-days")
        if Q["outliers"].empty:
            st.info("None found.")
        else:
            st.dataframe(
                Q["outliers"].head(200),
                hide_index=True,
                width="stretch",
                column_config={
                    "product": "Article",
                    "period": st.column_config.DateColumn("Day"),
                    "qty": st.column_config.NumberColumn("Units", format="%,.0f"),
                    "typical": st.column_config.NumberColumn("Typical", format="%,.1f"),
                    "ratio": st.column_config.NumberColumn("× typical", format="%.1f"),
                    "kind": "Type",
                },
            )
