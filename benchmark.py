"""Benchmark forecastability: how well do simple methods forecast this portfolio?

For every period in the test window, each method forecasts `horizon` periods ahead using
only data available at that time. Errors are summarised as
  WAPE = sum |forecast - actual| / sum actual     (robust with zeros, volume-weighted)
  bias = sum (forecast - actual) / sum actual     (+ = over-forecasting)
A production model should beat the best of these; if it can't, the benchmark is the model.
"""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from analysis import group_rolling


def _rolling_past(
    panel: pd.DataFrame, lag: int, window: int, stat: str, min_periods: int
) -> pd.Series:
    """Rolling statistic over the `window` periods ending `lag` periods ago, per article."""
    shifted = panel.groupby("product", sort=False)["qty"].shift(lag)
    return group_rolling(
        shifted, panel["product"], window, stat, min_periods=min_periods
    )


def forecasts(panel: pd.DataFrame, freq: str, horizon: int) -> dict[str, pd.Series]:
    g = panel.groupby("product", sort=False)["qty"]
    if freq == "D":
        weeks_back = ceil(horizon / 7)
        return {
            "Naive": g.shift(horizon),
            "Same weekday last week": g.shift(7 * weeks_back),
            "Weekday average (4 wks)": pd.concat(
                [g.shift(7 * (weeks_back + i)) for i in range(4)], axis=1
            ).mean(axis=1),
            "Moving average (28 d)": _rolling_past(panel, horizon, 28, "mean", 7),
        }
    # year-based methods fall back to the moving average for articles with < 1 year of history
    ma = _rolling_past(panel, horizon, 8, "mean", 4)
    return {
        "Naive": g.shift(horizon),
        "Moving average (8 wks)": ma,
        "Same week last year": g.shift(52 * ceil(horizon / 52)).fillna(ma),
        "Last year × recent trend": _seasonal_trend(panel, horizon).fillna(ma),
    }


def _seasonal_trend(panel: pd.DataFrame, horizon: int) -> pd.Series:
    """Same week last year, scaled by how the last 8 known weeks compare with a year earlier."""
    g = panel.groupby("product", sort=False)["qty"]
    last_year = g.shift(52)
    recent = _rolling_past(panel, horizon, 8, "sum", 4)
    recent_ly = _rolling_past(panel, horizon + 52, 8, "sum", 4)
    factor = (recent / recent_ly.where(recent_ly > 0)).clip(0.25, 4)
    return last_year * factor


def backtest(panel: pd.DataFrame, freq: str, horizon: int, test_periods: int) -> dict:
    fc = forecasts(panel, freq, horizon)
    cutoff = panel["period"].max() - pd.Timedelta(
        days=(test_periods - 1) * (1 if freq == "D" else 7)
    )
    test = panel["period"] >= cutoff
    long = pd.concat(
        [
            panel.loc[test, ["product", "period", "qty", "promo"]].assign(
                method=name, forecast=f[test]
            )
            for name, f in fc.items()
        ],
        ignore_index=True,
    )

    # compare methods on the same rows: drop rows where any method has no forecast
    complete = (
        long["forecast"]
        .notna()
        .groupby([long["product"], long["period"]])
        .transform("all")
    )
    long = long[complete].assign(err=lambda d: d["forecast"] - d["qty"])
    long["abs_err"] = long["err"].abs()

    def summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        s = df.groupby(by).agg(
            actual=("qty", "sum"), abs_err=("abs_err", "sum"), err=("err", "sum")
        )
        s["wape"] = s["abs_err"] / s["actual"].where(s["actual"] > 0)
        s["bias"] = s["err"] / s["actual"].where(s["actual"] > 0)
        return s

    by_method = summary(long, ["method"]).sort_values("wape")
    best_method = by_method.index[0] if len(by_method) else None

    per = summary(long, ["product", "method"])["wape"].unstack("method")
    articles = pd.DataFrame(
        {
            "test_volume": long[long["method"] == best_method]
            .groupby("product")["qty"]
            .sum(),
            "best_method": per.idxmin(axis=1) if len(per) else pd.Series(dtype=object),
            "best_wape": per.min(axis=1),
            "portfolio_method_wape": per[best_method] if best_method else np.nan,
        }
    )
    articles["forecastability"] = (
        pd.cut(
            articles["best_wape"],
            [-np.inf, 0.2, 0.5, np.inf],
            labels=["easy", "medium", "hard"],
        )
        .astype(object)
        .where(articles["best_wape"].notna(), "no volume")
    )
    articles = (
        articles.sort_values("test_volume", ascending=False)
        .rename_axis("product")
        .reset_index()
    )

    # accuracy of the portfolio total (errors cancel when aggregated)
    agg = (
        long[long["method"] == best_method].groupby("period")[["qty", "forecast"]].sum()
    )
    total_wape = (agg["forecast"] - agg["qty"]).abs().sum() / max(
        agg["qty"].sum(), 1e-9
    )

    best = long[long["method"] == best_method]
    promo_err_share = best.loc[best["promo"], "abs_err"].sum() / max(
        best["abs_err"].sum(), 1e-9
    )
    return {
        "by_method": by_method.reset_index(),
        "articles": articles,
        "best_method": best_method,
        "portfolio": {
            "best_method": best_method,
            "wape": float(by_method["wape"].iloc[0]) if len(by_method) else np.nan,
            "bias": float(by_method["bias"].iloc[0]) if len(by_method) else np.nan,
            "total_wape": float(total_wape),
            "promo_error_share": float(promo_err_share),
            "promo_period_share": float(best["promo"].mean()) if len(best) else 0.0,
            "articles_tested": int(articles["best_wape"].notna().sum()),
        },
    }
