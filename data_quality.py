"""Data quality and latency: is the history complete, and when does data arrive?

Missing days are judged per account against that account's own normal week:
a day is only suspicious if the account usually sells on that weekday (5+ of the
same weekday 1-4 weeks before/after had sales) and it isn't a holiday.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import group_rolling

FLAGS = {
    "missing": "No sales at all on a normal trading day",
    "partial": "Under 70% of the usual articles and volume (partly loaded)",
    "low": "Under 30% of the usual volume",
    "spike": "Over 3× the usual volume, not explained by promotions",
}


# ------------------------------------------------------------- missing days


def account_days(
    sales: pd.DataFrame,
    holiday_dates: pd.Series,
    promo_days: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per account and day between its first and last row, with a flag.
    promo_days (product, date) lets promo-driven peaks pass as normal."""
    s = sales.assign(sold=sales["product"].where(sales["qty"] != 0))
    if promo_days is not None and len(promo_days):
        s = s.merge(
            promo_days.assign(on_promo=True), on=["product", "date"], how="left"
        )
        s["promo_qty"] = s["qty"].where(s["on_promo"].fillna(False).astype(bool), 0.0)
    else:
        s["promo_qty"] = 0.0
    d = (
        s.groupby(["location", "date"])
        .agg(
            qty=("qty", "sum"),
            articles=("sold", "nunique"),
            promo_qty=("promo_qty", "sum"),
        )
        .reset_index()
    )
    span = d.groupby("location")["date"].agg(["min", "max"])
    grid = pd.concat(
        [
            pd.DataFrame(
                {"location": loc, "date": pd.date_range(r["min"], r["max"], freq="D")}
            )
            for loc, r in span.iterrows()
        ],
        ignore_index=True,
    )
    d = grid.merge(d, on=["location", "date"], how="left").fillna(
        {"qty": 0.0, "articles": 0, "promo_qty": 0.0}
    )
    d = d.sort_values(["location", "date"]).reset_index(drop=True)

    g = d.groupby("location", sort=False)
    weeks = (-4, -3, -2, -1, 1, 2, 3, 4)
    ref_qty = pd.concat([g["qty"].shift(7 * k) for k in weeks], axis=1)
    ref_art = pd.concat([g["articles"].shift(7 * k) for k in weeks], axis=1)
    d["usual_qty"] = ref_qty.median(axis=1)
    d["usual_articles"] = ref_art.median(axis=1)
    normally_open = (ref_art > 0).sum(axis=1) >= 5
    eligible = normally_open & ~d["date"].isin(holiday_dates) & (d["usual_qty"] > 0)

    ratio = d["qty"] / d["usual_qty"].where(d["usual_qty"] > 0)
    d["ratio"] = ratio
    promo_driven = d["promo_qty"] > 0.5 * d["qty"].where(d["qty"] > 0)
    d["flag"] = np.select(
        [
            eligible & (d["articles"] == 0),
            eligible & (d["articles"] < 0.7 * d["usual_articles"]) & (ratio < 0.7),
            eligible & (ratio < 0.3),
            eligible & (ratio > 3) & ~promo_driven,
        ],
        ["missing", "partial", "low", "spike"],
        default="",
    )
    return d


def account_summary(days: pd.DataFrame, silent_after: int = 7) -> pd.DataFrame:
    end = days["date"].max()
    trading = days[days["articles"] > 0]
    acc = days.groupby("location").agg(first_day=("date", "min"), volume=("qty", "sum"))
    acc["last_sale"] = trading.groupby("location")["date"].max()
    acc["trading_days"] = trading.groupby("location").size()
    for f in FLAGS:
        acc[f] = (days["flag"] == f).groupby(days["location"]).sum()
    acc["days_silent"] = (end - acc["last_sale"]).dt.days
    acc["status"] = np.where(
        (acc["days_silent"] >= silent_after) & (acc["trading_days"] >= 28),
        "silent " + acc["days_silent"].astype(str) + " days",
        "active",
    )
    acc["volume_share"] = acc["volume"] / max(acc["volume"].sum(), 1e-9)
    return acc.sort_values("volume", ascending=False).reset_index()


def system_outages(days: pd.DataFrame, min_share: float = 0.5) -> pd.DataFrame:
    """Days where at least `min_share` of accounts that normally trade are missing."""
    judged = days[days["usual_qty"] > 0]
    # a boolean sum, not a lambda agg: that keeps the str dtype when `judged` is empty
    by_day = pd.DataFrame(
        {
            "accounts": judged.groupby("date").size(),
            "missing": (judged["flag"] == "missing").groupby(judged["date"]).sum(),
        }
    )
    by_day = by_day[
        (by_day["accounts"] >= 2)
        & (by_day["missing"] >= min_share * by_day["accounts"])
    ]
    return by_day.reset_index()


# ------------------------------------------------------------------ latency


def latency(q: pd.DataFrame, window: int = 90, mature: int = 14) -> dict:
    """q: date (local sales day), lag_hours (created_at - orderDate, 6h buckets), n_rows.
    Measured on sales days old enough that their data should be complete."""
    if q.empty:
        return {
            "curve": pd.DataFrame(columns=["day", "share"]),
            "bulk_loaded": False,
            "k50": None,
            "k80": None,
            "k95": None,
            "last_day": None,
        }
    end = q["date"].max()
    w = q[
        (q["date"] > end - pd.Timedelta(days=window + mature))
        & (q["date"] <= end - pd.Timedelta(days=mature))
    ]
    total = w["n_rows"].sum()
    days = np.arange(0, 31)
    share = [
        (
            w.loc[w["lag_hours"] < 24 * (k + 1), "n_rows"].sum() / total
            if total
            else np.nan
        )
        for k in days
    ]
    curve = pd.DataFrame({"day": days, "share": share})
    median_lag = np.repeat(
        w["lag_hours"].to_numpy(), w["n_rows"].to_numpy().astype(int)
    )
    median_lag = float(np.median(median_lag)) / 24 if len(median_lag) else np.nan

    def first(p: float):
        hit = curve.loc[curve["share"] >= p, "day"]
        return int(hit.iloc[0]) if len(hit) else None

    return {
        "curve": curve,
        "bulk_loaded": bool(median_lag > 30),
        "median_lag_days": median_lag,
        "k50": first(0.5),
        "k80": first(0.8),
        "k95": first(0.95),
        "last_day": end,
    }


def arrivals(q: pd.DataFrame, backfill_days: int = 30) -> pd.DataFrame:
    """Rows per arrival day; rows arriving more than `backfill_days` late count as backfill."""
    a = q.assign(
        arrival=(q["date"] + pd.to_timedelta(q["lag_hours"], unit="h")).dt.normalize(),
        kind=np.where(
            q["lag_hours"] > 24 * backfill_days, "backfill / re-import", "regular"
        ),
    )
    return a.groupby(["arrival", "kind"], as_index=False)["n_rows"].sum()


# ------------------------------------------------------------ rewrites etc.


def changes(q: pd.DataFrame) -> dict:
    tot = q[
        ["n_rows", "negative_rows", "zero_rows", "updated_rows", "deleted_rows"]
    ].sum()
    n = max(tot["n_rows"], 1)
    monthly = (
        q.assign(month=q["date"].dt.to_period("M").dt.start_time)
        .groupby("month")[["n_rows", "updated_rows", "deleted_rows", "negative_rows"]]
        .sum()
    )
    for c in ["updated_rows", "deleted_rows", "negative_rows"]:
        monthly[c.replace("_rows", "_share")] = monthly[c] / monthly["n_rows"].where(
            monthly["n_rows"] > 0
        )
    return {
        "rows": int(tot["n_rows"]),
        "negative": tot["negative_rows"] / n,
        "zero": tot["zero_rows"] / n,
        "updated": tot["updated_rows"] / n,
        "deleted": tot["deleted_rows"] / n,
        "monthly": monthly.reset_index(),
    }


def outliers(
    daily: pd.DataFrame, factor: float = 8, min_excess: float = 10
) -> pd.DataFrame:
    """Article-days far above the article's typical level (outside promos), and net-negative days."""
    typical = group_rolling(
        daily["qty"], daily["product"], 57, "mean", center=True, min_periods=14
    )
    d = daily.assign(typical=typical, ratio=daily["qty"] / typical.where(typical > 0))
    spike = (
        (d["qty"] > factor * d["typical"])
        & (d["qty"] - d["typical"] >= min_excess)
        & ~d["promo"]
    )
    out = d[spike | (d["qty"] < 0)]
    out = out.assign(kind=np.where(out["qty"] < 0, "net negative", "spike"))
    return out[["product", "period", "qty", "typical", "ratio", "kind"]].sort_values(
        "ratio", ascending=False
    )


# ------------------------------------------------------------------- main


def analyse(
    sales: pd.DataFrame,
    daily: pd.DataFrame,
    quality: pd.DataFrame | None,
    holiday_dates: pd.Series,
    today: pd.Timestamp,
) -> dict:
    promo_days = daily.loc[daily["promo"], ["product", "period"]].rename(
        columns={"period": "date"}
    )
    days = account_days(sales, holiday_dates, promo_days)
    accounts = account_summary(days)
    lat = latency(quality) if quality is not None else None
    last_day = sales["date"].max()
    flagged = days[days["flag"] != ""]
    volume_on_flagged = days.loc[
        days["flag"].isin(["missing", "partial"]), "usual_qty"
    ].sum()
    return {
        "days": days,
        "flagged": flagged[
            [
                "location",
                "date",
                "flag",
                "qty",
                "usual_qty",
                "articles",
                "usual_articles",
            ]
        ],
        "accounts": accounts,
        "outages": system_outages(days),
        "latency": lat,
        "arrivals": arrivals(quality) if quality is not None else None,
        "changes": changes(quality) if quality is not None else None,
        "outliers": outliers(daily),
        "summary": {
            "last_day": last_day,
            "days_old": int((today.normalize() - last_day).days),
            "flag_counts": flagged["flag"].value_counts().to_dict(),
            "missing_volume_share": volume_on_flagged / max(days["qty"].sum(), 1e-9),
            "silent_accounts": accounts[accounts["status"] != "active"],
        },
    }
