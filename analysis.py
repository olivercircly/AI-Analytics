"""Promo analytics without any Streamlit or DB dependency, so it can be unit-tested.

Conventions
-----------
lag = (period of sales) - (period of recorded promo)
  lag < 0  sales happen BEFORE the recorded promo (e.g. distributor sell-in for a store promo)
  lag > 0  sales happen AFTER it (e.g. promo dates are announcement dates, reporting delay)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Settings:
    freq: str = "W-MON"  # "D" (daily) or "W-MON" (weeks starting Monday)
    by_location: bool = False
    max_lag: int = 3  # shifts tested in each direction, in periods
    baseline_window: int = 13  # centred rolling median over non-promo periods
    min_events: int = 3  # promo starts needed before trusting a per-series shift
    low_dependency: float = 0.10
    high_dependency: float = 0.30

    @property
    def keys(self) -> list[str]:
        return ["product", "location"] if self.by_location else ["product"]

    @property
    def period_length(self) -> pd.Timedelta:
        return pd.Timedelta(days=1 if self.freq == "D" else 7)


@dataclass
class Result:
    settings: Settings
    panel: pd.DataFrame  # one row per series x period
    metrics: pd.DataFrame  # one row per series
    lag_scores: pd.DataFrame  # series x lag
    pooled_lag_scores: pd.Series  # lag -> score over the whole portfolio
    event_profile: pd.DataFrame  # average uplift around promo starts
    portfolio: dict


def to_period(dates: pd.Series, freq: str) -> pd.Series:
    dates = pd.to_datetime(dates).dt.normalize()
    if freq == "D":
        return dates
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


# --------------------------------------------------------------------------- panel


def _expand_promos(promos: pd.DataFrame, s: Settings, lo, hi) -> pd.DataFrame:
    """One row per promo day, clipped to the sales history."""
    p = promos.dropna(subset=["promo_start", "promo_end"]).copy()
    p["promo_start"] = p["promo_start"].clip(lower=lo)
    p["promo_end"] = p["promo_end"].clip(upper=hi)
    p = p[p["promo_end"] >= p["promo_start"]]
    if p.empty:
        return p.assign(day=pd.NaT, period=pd.NaT)
    p["day"] = [
        pd.date_range(a, b, freq="D") for a, b in zip(p["promo_start"], p["promo_end"])
    ]
    p = p.explode("day")
    p["day"] = pd.to_datetime(p["day"])
    p["period"] = to_period(p["day"], s.freq)
    return p


def _attach_promos(
    panel: pd.DataFrame, promos: pd.DataFrame, s: Settings
) -> pd.DataFrame:
    lo, hi = panel["period"].min(), panel[
        "period"
    ].max() + s.period_length - pd.Timedelta(days=1)
    p = _expand_promos(promos, s, lo, hi)
    if p.empty:
        return panel.assign(promo_days=0.0)

    def count(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return (
            df.groupby(keys + ["period"])["day"]
            .nunique()
            .rename("promo_days")
            .reset_index()
        )

    if not s.by_location:
        panel = panel.merge(count(p, ["product"]), on=["product", "period"], how="left")
    else:
        local = count(p[p["location"].notna()], ["product", "location"])
        everywhere = count(p[p["location"].isna()], ["product"]).rename(
            columns={"promo_days": "all_loc"}
        )
        panel = panel.merge(
            local, on=["product", "location", "period"], how="left"
        ).merge(everywhere, on=["product", "period"], how="left")
        panel["promo_days"] = panel[["promo_days", "all_loc"]].max(axis=1)
        panel = panel.drop(columns="all_loc")
    panel["promo_days"] = panel["promo_days"].fillna(0.0)
    return panel


def build_panel(sales: pd.DataFrame, promos: pd.DataFrame, s: Settings) -> pd.DataFrame:
    keys = s.keys
    agg = (
        sales.assign(period=to_period(sales["date"], s.freq))
        .groupby(keys + ["period"], as_index=False)["qty"]
        .sum()
    )

    # complete grid: missing periods are zero sales, but only between first and last sale
    periods = pd.date_range(agg["period"].min(), agg["period"].max(), freq=s.freq)
    grid = (
        agg[keys]
        .drop_duplicates()
        .merge(pd.DataFrame({"period": periods}), how="cross")
    )
    panel = grid.merge(agg, on=keys + ["period"], how="left").fillna({"qty": 0.0})
    active = (
        agg[agg["qty"] > 0]
        .groupby(keys)["period"]
        .agg(first="min", last="max")
        .reset_index()
    )
    panel = panel.merge(active, on=keys)
    panel = panel[panel["period"].between(panel["first"], panel["last"])].drop(
        columns=["first", "last"]
    )

    panel = _attach_promos(panel, promos, s)
    panel["promo"] = panel["promo_days"] > 0
    panel["series"] = panel[keys].astype(str).agg(" @ ".join, axis=1)
    return panel.sort_values(keys + ["period"]).reset_index(drop=True)


# --------------------------------------------------------------------- baseline


def add_baseline(panel: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Rolling median of 'clean' periods. Periods near a promo are excluded too,
    otherwise a shifted promo effect would leak into the baseline."""
    keys = s.keys
    guard = 2 * s.max_lag + 1
    near_promo = (
        panel.groupby(keys, sort=False)["promo"]
        .transform(
            lambda x: x.astype(float).rolling(guard, center=True, min_periods=1).max()
        )
        .astype(bool)
    )
    panel = panel.assign(clean=panel["qty"].where(~near_promo))
    min_obs = max(2, s.baseline_window // 3)
    panel["baseline"] = panel.groupby(keys, sort=False)["clean"].transform(
        lambda x: x.rolling(s.baseline_window, center=True, min_periods=min_obs)
        .median()
        .interpolate(limit_direction="both")
    )
    level = panel.groupby(keys, sort=False)["qty"].transform("mean").replace(0, np.nan)
    panel["uplift"] = panel["qty"] - panel["baseline"]
    panel["norm_uplift"] = (
        panel["uplift"] / level
    )  # in multiples of the series' average sales
    return panel.drop(columns="clean")


# ------------------------------------------------------------------ shift tests


def lag_scores(panel: pd.DataFrame, s: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """Score(k) = mean uplift when promo(t-k) is on minus when it is off."""
    keys = s.keys
    g = panel.groupby(keys, sort=False)["promo"]
    u = panel["norm_uplift"]
    per_series, pooled = {}, {}
    for k in range(-s.max_lag, s.max_lag + 1):
        on = g.shift(k, fill_value=False).astype(bool)
        m = (
            panel[keys]
            .assign(on=on, u=u)
            .groupby(keys + ["on"])["u"]
            .mean()
            .unstack("on")
            .reindex(columns=[False, True])
        )
        per_series[k] = m[True] - m[False]
        pooled[k] = u[on].mean() - u[~on].mean()
    return pd.DataFrame(per_series), pd.Series(pooled, name="score").rename_axis("lag")


def promo_starts(
    panel: pd.DataFrame, s: Settings, quiet_before: int
) -> tuple[pd.Series, pd.Series]:
    """All promo starts, and 'isolated' ones with no promo in the preceding periods."""
    g = panel.groupby(s.keys, sort=False)["promo"]
    starts = panel["promo"] & ~g.shift(1, fill_value=False).astype(bool)
    recent = g.transform(
        lambda x: x.shift(1, fill_value=False)
        .astype(float)
        .rolling(quiet_before, min_periods=1)
        .max()
    ).astype(bool)
    return starts, starts & ~recent


def event_profile(panel: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Average normalised uplift from `pre` periods before to `post` after each promo start."""
    pre, post = s.max_lag + 2, s.max_lag + 4
    _, isolated = promo_starts(panel, s, quiet_before=pre)
    g = panel.groupby(s.keys, sort=False)["norm_uplift"]
    frames = [
        pd.DataFrame({"offset": d, "u": g.shift(-d)[isolated].to_numpy()})
        for d in range(-pre, post + 1)
    ]
    long = pd.concat(frames, ignore_index=True).dropna()
    if long.empty:
        return pd.DataFrame(columns=["offset", "mean", "lo", "hi", "events"])
    agg = long.groupby("offset")["u"].agg(["mean", "std", "count"])
    se = (agg["std"] / np.sqrt(agg["count"])).fillna(0)
    return pd.DataFrame(
        {
            "offset": agg.index,
            "mean": agg["mean"],
            "lo": agg["mean"] - 1.96 * se,
            "hi": agg["mean"] + 1.96 * se,
            "events": agg["count"],
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------- dependency


def series_metrics(
    panel: pd.DataFrame, lags: pd.DataFrame, portfolio_shift: int, s: Settings
) -> pd.DataFrame:
    keys = s.keys
    g = panel.groupby(keys, sort=False)
    starts, _ = promo_starts(panel, s, quiet_before=1)
    events = starts.groupby([panel[k] for k in keys]).sum()

    # per-series shift, trusted only with enough promo events
    best = lags.dropna(how="all").idxmax(axis=1).reindex(events.index)
    best[events < s.min_events] = np.nan

    # 'effective' promo = recorded promo moved by the series' own shift
    # (portfolio shift as fallback for series with too few events)
    row_lag = (
        pd.MultiIndex.from_frame(panel[keys])
        if len(keys) > 1
        else pd.Index(panel[keys[0]])
    )
    row_lag = best.fillna(portfolio_shift).reindex(row_lag).to_numpy()
    eff = pd.Series(False, index=panel.index)
    for k in np.unique(row_lag):
        mask = row_lag == k
        eff[mask] = g["promo"].shift(int(k), fill_value=False).astype(bool)[mask]

    has_base = panel["baseline"].notna()
    p = panel.assign(
        promo_qty=panel["qty"].where(panel["promo"], 0.0),
        eff_qty=panel["qty"].where(eff, 0.0),
        incr_qty=panel["uplift"].clip(lower=0).where(eff & has_base, 0.0),
        eff_sales=panel["qty"].where(eff & has_base, 0.0),
        eff_base=panel["baseline"].where(eff & has_base, 0.0),
    )
    m = (
        p.groupby(keys + ["series"])
        .agg(
            volume=("qty", "sum"),
            periods=("qty", "size"),
            promo_period_share=("promo", "mean"),
            promo_qty=("promo_qty", "sum"),
            eff_qty=("eff_qty", "sum"),
            incr_qty=("incr_qty", "sum"),
            eff_sales=("eff_sales", "sum"),
            eff_base=("eff_base", "sum"),
        )
        .reset_index(level="series")
    )
    m["promo_events"] = events.reindex(m.index).astype(int)

    vol = m["volume"].replace(0, np.nan)
    m["promo_volume_share"] = m["promo_qty"] / vol  # as recorded
    m["promo_volume_share_adj"] = m["eff_qty"] / vol  # shift-adjusted
    m["incremental_share"] = m["incr_qty"] / vol  # volume that exists because of promos
    m["promo_lift"] = m["eff_sales"] / m["eff_base"].replace(0, np.nan)
    m["best_lag"] = best.reindex(m.index)
    m["lag_gain"] = (lags.max(axis=1) - lags[0]).reindex(m.index)

    m["dependency"] = pd.cut(
        m["incremental_share"],
        [-np.inf, s.low_dependency, s.high_dependency, np.inf],
        labels=["low", "medium", "high"],
    ).astype(object)
    m.loc[m["promo_events"] == 0, "dependency"] = "no promos"
    return m.reset_index().sort_values("volume", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------- main


def analyse(sales: pd.DataFrame, promos: pd.DataFrame, s: Settings) -> Result:
    if sales.empty:
        raise ValueError("No sales rows for this customer and date range.")
    panel = add_baseline(build_panel(sales, promos, s), s)
    lags, pooled = lag_scores(panel, s)
    shift = int(pooled.idxmax()) if pooled.notna().any() else 0
    metrics = series_metrics(panel, lags, shift, s)

    total = metrics["volume"].sum()
    portfolio = {
        "series": len(metrics),
        "promoted_series": int((metrics["promo_events"] > 0).sum()),
        "promo_volume_share": float(metrics["promo_qty"].sum() / total),
        "promo_volume_share_adj": float(metrics["eff_qty"].sum() / total),
        "incremental_share": float(metrics["incr_qty"].sum() / total),
        "high_dependency_volume": metrics.loc[
            metrics["dependency"] == "high", "volume"
        ].sum()
        / total,
        "shift": shift,
        "shifted_volume_share": metrics.loc[
            metrics["best_lag"].fillna(0) != 0, "volume"
        ].sum()
        / max(metrics.loc[metrics["best_lag"].notna(), "volume"].sum(), 1e-9),
        "tested_series": int(metrics["best_lag"].notna().sum()),
        "shift_gain": (
            float(pooled.max() - pooled.get(0, np.nan)) if pooled.notna().any() else 0.0
        ),
    }
    return Result(s, panel, metrics, lags, pooled, event_profile(panel, s), portfolio)
