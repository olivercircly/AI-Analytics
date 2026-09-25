"""Weekday and yearly seasonality per article and for the whole portfolio.

Both use the same recipe on a multiplicative decomposition:
  ratio     = sales / trend             (trend = centred moving average over one season)
  index     = average ratio per season position (weekday or week of year)
  remainder = ratio - index
  strength  = max(0, 1 - Var(remainder) / Var(ratio))      (Hyndman-style, 0 = none, 1 = pure seasonality)
Promo periods are left out so promotions don't masquerade as seasonality.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd

from analysis import group_rolling

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _decompose(
    df: pd.DataFrame, season: int, position: pd.Series, min_periods: int
) -> pd.DataFrame:
    trend = group_rolling(
        df["qty"], df["product"], season, "mean", center=True, min_periods=min_periods
    )
    out = df.assign(trend=trend, pos=position)
    out = out[(out["trend"] > 0) & ~out["promo"]]
    return out.assign(ratio=out["qty"] / out["trend"])


def _strength(d: pd.DataFrame, index: pd.DataFrame) -> pd.Series:
    d = d.join(index.stack().rename("index"), on=["product", "pos"])
    d = d.assign(rem=d["ratio"] - d["index"])
    agg = d.groupby("product").agg(var_all=("ratio", "var"), var_rem=("rem", "var"))
    return (1 - agg["var_rem"] / agg["var_all"]).clip(0, 1)


def weekday(daily: pd.DataFrame, min_weeks: int = 8) -> tuple[pd.Series, pd.DataFrame]:
    """daily: product, period (date), qty, promo. Returns strength and a product x weekday index."""
    d = _decompose(daily, 7, daily["period"].dt.dayofweek, min_periods=7)
    index = (
        d.groupby(["product", "pos"])["ratio"]
        .mean()
        .unstack("pos")
        .reindex(columns=range(7))
    )
    index = index.div(index.mean(axis=1), axis=0)
    strength = _strength(d, index)
    enough = d.groupby("product").size() >= 7 * min_weeks
    return strength.where(enough), index


def yearly(
    weekly: pd.DataFrame, min_years: float = 2.0, smooth: int = 5
) -> tuple[pd.Series, pd.DataFrame]:
    """weekly: product, period (Monday), qty, promo. Returns strength and a product x week-of-year index."""
    woy = weekly["period"].dt.isocalendar().week.clip(upper=52).astype(int)
    d = _decompose(weekly, 52, woy, min_periods=26)
    raw = (
        d.groupby(["product", "pos"])["ratio"]
        .mean()
        .unstack("pos")
        .reindex(columns=range(1, 53))
    )

    # circular smoothing across weeks: single week-of-year means are noisy with only a few years
    arr = raw.to_numpy(dtype=float)
    half = smooth // 2
    padded = np.concatenate([arr[:, -half:], arr, arr[:, :half]], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        smoothed = np.nanmean(
            np.lib.stride_tricks.sliding_window_view(padded, smooth, axis=1), axis=-1
        )
    index = pd.DataFrame(smoothed, index=raw.index, columns=raw.columns)
    index = index.div(index.mean(axis=1), axis=0)

    strength = _strength(d, index)
    span = weekly.groupby("product")["period"].agg(["min", "max"])
    span_weeks = (span["max"] - span["min"]).dt.days / 7 + 1
    return strength.where(span_weeks.reindex(strength.index) >= 52 * min_years), index


def week_to_month(week: int) -> str:
    return MONTHS[date.fromisocalendar(2025, int(week), 4).month - 1]


def classify(strength: pd.Series) -> pd.Series:
    return (
        pd.cut(strength, [-0.01, 0.3, 0.6, 1.0], labels=["weak", "moderate", "strong"])
        .astype(object)
        .where(strength.notna(), "too short")
    )


def analyse(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    wd_strength, wd_index = weekday(daily)
    yr_strength, yr_index = yearly(weekly)

    volume = daily.groupby("product")["qty"].sum()
    per_article = pd.DataFrame({"volume": volume})
    per_article["weekday_strength"] = wd_strength
    per_article["yearly_strength"] = yr_strength
    per_article["weekday_class"] = classify(per_article["weekday_strength"])
    per_article["yearly_class"] = classify(per_article["yearly_strength"])
    valid = yr_index.dropna(how="all")
    per_article["amplitude"] = (valid.max(axis=1) - valid.min(axis=1)).where(
        yr_strength.notna()
    )
    peak_week = valid.idxmax(axis=1)
    per_article["peak_month"] = peak_week.map(week_to_month).where(yr_strength.notna())
    per_article = (
        per_article.sort_values("volume", ascending=False)
        .rename_axis("product")
        .reset_index()
    )

    # portfolio = all articles summed, analysed as one series
    total_d = daily.groupby("period", as_index=False).agg(
        qty=("qty", "sum"), promo=("promo", "mean")
    )
    total_w = weekly.groupby("period", as_index=False).agg(
        qty=("qty", "sum"), promo=("promo", "mean")
    )
    total_d = total_d.assign(product="Portfolio", promo=False)
    total_w = total_w.assign(product="Portfolio", promo=False)
    pwd_s, pwd_i = weekday(total_d)
    pyr_s, pyr_i = yearly(total_w)

    overlay = total_w.assign(
        year=total_w["period"].dt.isocalendar().year.astype(int),
        week=total_w["period"].dt.isocalendar().week.astype(int),
    )

    def share(col: str, cls: str) -> float:
        return per_article.loc[per_article[col] == cls, "volume"].sum() / max(
            per_article["volume"].sum(), 1e-9
        )

    yr_prof = pyr_i.iloc[0] if len(pyr_i) else pd.Series(dtype=float)
    return {
        "per_article": per_article,
        "weekday_profile": (
            pd.DataFrame({"weekday": WEEKDAYS, "index": pwd_i.iloc[0].to_numpy()})
            if len(pwd_i)
            else None
        ),
        "yearly_profile": (
            yr_prof.rename_axis("week").reset_index(name="index")
            if len(yr_prof)
            else None
        ),
        "overlay": overlay[["year", "week", "qty"]],
        "weekday_index": wd_index,
        "yearly_index": yr_index,
        "portfolio": {
            "weekday_strength": float(pwd_s.iloc[0]) if len(pwd_s) else np.nan,
            "yearly_strength": float(pyr_s.iloc[0]) if len(pyr_s) else np.nan,
            "peak_month": (
                week_to_month(yr_prof.idxmax())
                if yr_prof.notna().any() and pd.notna(pyr_s.iloc[0])
                else None
            ),
            "amplitude": (
                float(yr_prof.max() - yr_prof.min())
                if yr_prof.notna().any()
                else np.nan
            ),
            "strong_weekday_volume": share("weekday_class", "strong"),
            "strong_yearly_volume": share("yearly_class", "strong"),
            "seasonal_yearly_volume": share("yearly_class", "strong")
            + share("yearly_class", "moderate"),
        },
    }
