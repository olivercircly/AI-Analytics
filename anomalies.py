"""Anomalies in a single article's history: how it developed and what looks wrong.

Everything is judged against the article's own expected level: a centred rolling
mean over normal periods (no promos, no long zero runs, spikes capped), times a
weekday factor for daily data. A mean rather than a median, because for slow movers
most days are zero and the median would expect nothing. Holidays and weekdays the article normally doesn't sell on (e.g.
Sundays) have no expectation, so they are never flagged.

  spike / drop   a period far above / below expected, outside (shifted) promos
  gap            a run of zero periods that should have sold `min_expected` units or more
  stopped        no sales since a date, although the usual rate says there should be
  level shift    a lasting, statistically clear step change in the deseasonalised
                 weekly level, attributed to the account that changed most
  flat promo     a recorded promo without visible uplift per promo day
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import analysis as an
import seasonality as se

KINDS = {
    "spike": "Far above the expected level, no promo recorded",
    "drop": "Far below the expected level (but not zero)",
    "gap": "No sales for a stretch that should have sold",
    "stopped": "No sales since this date, although there should be",
    "level shift": "Lasting change in the weekly sales level (seasonality removed)",
    "flat promo": "Recorded promo without visible uplift",
}
EVENT_COLUMNS = ["kind", "start", "end", "account", "qty", "expected"]


# ------------------------------------------------------------------ expectation


def _rolling_baseline(qty: pd.Series, normal: pd.Series, window: int) -> pd.Series:
    """Centred rolling mean over `normal` periods, with values above the rolling 90th
    percentile capped so single spikes don't lift the expectation around them."""
    x = qty.where(normal)
    roll = dict(window=window, center=True, min_periods=max(2, window // 3))
    capped = x.clip(upper=x.rolling(**roll).quantile(0.9))
    rolled = capped.rolling(**roll).mean()
    return rolled.interpolate(limit_area="inside").ffill().bfill()


def _weekday_expectation(
    panel: pd.DataFrame,
    baseline: pd.Series,
    freq: str,
    promo: pd.Series,
    holiday: pd.Series,
) -> pd.Series:
    """Baseline x weekday factor (daily); NaN where no sales are expected."""
    if freq != "D":
        return baseline.where(baseline > 0)
    ok = ~promo & ~holiday & (baseline > 0)
    dow = panel["period"].dt.dayofweek
    factor = panel["qty"][ok].groupby(dow[ok]).sum() / baseline[ok].groupby(dow[ok]).sum()
    factor = factor.where(factor >= 0.05)  # weekdays it (almost) never sells on
    exp = baseline * panel["period"].dt.dayofweek.map(factor).to_numpy()
    return exp.where(~holiday & (exp > 0))


def _zero_run_mask(qty: pd.Series, expected: pd.Series, min_len: int) -> pd.Series:
    """Periods inside runs of at least `min_len` judged zero periods (unjudged periods
    such as Sundays neither count nor break a run)."""
    judged = expected.notna()
    zero = (qty == 0)[judged]
    run_id = (zero != zero.shift()).cumsum()
    size = zero.groupby(run_id).transform("sum")
    long_run = zero & (size >= min_len)
    if not long_run.any():
        return pd.Series(False, index=qty.index)
    bounds = long_run.groupby(run_id[long_run]).apply(lambda r: (r.index[0], r.index[-1]))
    mask = pd.Series(False, index=qty.index)
    for a, b in bounds:
        mask.loc[a:b] = True
    return mask


def expected_level(
    panel: pd.DataFrame,
    freq: str,
    window: int,
    promo: pd.Series,
    holiday_dates: pd.Series | None = None,
) -> pd.Series:
    """Two passes: the second baseline ignores long zero runs, so a stock-out doesn't
    drag its own expectation down to zero and hide itself."""
    holiday = (
        panel["period"].isin(holiday_dates)
        if holiday_dates is not None and freq == "D"
        else pd.Series(False, index=panel.index)
    )
    near_promo = (
        promo.astype(float).rolling(3, center=True, min_periods=1).max().astype(bool)
    )
    normal = ~near_promo & ~holiday
    first = _weekday_expectation(
        panel, _rolling_baseline(panel["qty"], normal, window), freq, promo, holiday
    )
    stretch = _zero_run_mask(panel["qty"], first, min_len=3)
    base = _rolling_baseline(panel["qty"], normal & ~stretch, window)
    return _weekday_expectation(panel, base, freq, promo, holiday)


def _promo_mask(panel: pd.DataFrame, shift: int) -> pd.Series:
    """Recorded promo periods plus the same periods moved by the detected shift."""
    shifted = panel["promo"].shift(shift, fill_value=False).astype(bool)
    return panel["promo"] | shifted


def _robust_scale(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 10:
        return 1.0
    return max(1.0, 1.4826 * float((r - r.median()).abs().median()))


# ------------------------------------------------------------- point anomalies


def spikes_and_drops(
    panel: pd.DataFrame,
    expected: pd.Series,
    promo: pd.Series,
    z_spike: float = 5,
    z_drop: float = 4,
    min_excess: float = 5,
) -> pd.DataFrame:
    """Periods far from expected, scored as Poisson residuals scaled by the article's
    own (robust) noise level, so a noisy article needs a bigger jump to be flagged."""
    r = (panel["qty"] - expected) / np.sqrt(expected.clip(lower=1))
    z = r / _robust_scale(r.where(~promo))
    spike = (
        (z >= z_spike)
        & (panel["qty"] >= 2 * expected)
        & (panel["qty"] - expected >= min_excess)
        & ~promo
    )
    drop = (panel["qty"] > 0) & (z <= -z_drop) & (panel["qty"] <= 0.5 * expected)
    hit = spike | drop
    return pd.DataFrame(
        {
            "kind": np.where(spike[hit], "spike", "drop"),
            "start": panel.loc[hit, "period"],
            "end": panel.loc[hit, "period"],
            "account": None,
            "qty": panel.loc[hit, "qty"],
            "expected": expected[hit],
        }
    )


def gaps(
    panel: pd.DataFrame, expected: pd.Series, min_expected: float = 10
) -> pd.DataFrame:
    """Runs of zero sales on periods where sales were expected, worth at least
    `min_expected` units in total."""
    judged = panel.loc[expected.notna(), ["period", "qty"]].assign(
        expected=expected[expected.notna()]
    )
    zero = judged["qty"] == 0
    if not zero.any():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    run_id = (zero != zero.shift()).cumsum()[zero]
    runs = (
        judged[zero]
        .groupby(run_id)
        .agg(start=("period", "min"), end=("period", "max"), expected=("expected", "sum"))
        .reset_index(drop=True)
    )
    runs = runs[runs["expected"] >= min_expected]
    return runs.assign(kind="gap", account=None, qty=0.0)[EVENT_COLUMNS]


# -------------------------------------------------------------- level shifts


def deseasonalise(weekly: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Weekly qty divided by the yearly index, whenever there is enough history (two
    years) to estimate one. A weak pattern has an index close to 1, so this is harmless."""
    y = weekly.set_index("period")["qty"].astype(float)
    strength, index = se.yearly(weekly.assign(product="x"))
    s = strength.get("x", np.nan)
    if pd.isna(s):
        return y, False
    woy = y.index.isocalendar().week.clip(upper=52).astype(int)
    factor = index.loc["x"].reindex(woy).to_numpy()
    return y / np.where(factor > 0.05, factor, np.nan), True


def level_shifts(
    weekly: pd.DataFrame,
    by_account: pd.DataFrame,
    exclude: pd.Series | None = None,
    window: int = 10,
    min_ratio: float = 1.5,
    min_t: float = 4,
    min_level: float = 3,
) -> pd.DataFrame:
    """Weeks where the mean of the next `window` normal weeks differs from the mean of
    the previous `window` by at least `min_ratio` AND clearly beyond noise (Welch
    t >= `min_t`), after removing yearly seasonality. Promo weeks and `exclude` weeks
    (e.g. inside a gap) are left out. Each shift is attributed to the account whose
    sales changed most."""
    y, _ = deseasonalise(weekly)
    skip = weekly.set_index("period")["promo"]
    if exclude is not None:
        skip = skip | exclude.reindex(skip.index, fill_value=False)
    y = y.where(~skip)
    need = max(3, window * 3 // 4)

    def side(v: pd.Series, stat: str) -> pd.Series:
        return getattr(v.rolling(window, min_periods=need), stat)()

    rev = y[::-1]
    before = side(y, "mean").shift(1)
    after = side(rev, "mean")[::-1]
    se2 = (side(y, "var") / side(y, "count")).shift(1) + (
        side(rev, "var") / side(rev, "count")
    )[::-1]
    t_stat = (after - before) / np.sqrt(se2.where(se2 > 0))
    log_ratio = np.log((after + 0.5) / (before + 0.5))
    strength = log_ratio.abs().where(
        (np.maximum(before, after) >= min_level)
        & (log_ratio.abs() >= np.log(min_ratio))
        & (t_stat.abs() >= min_t)
    )
    rows = []
    while strength.notna().any():
        t = strength.idxmax()
        rows.append((t, before[t], after[t]))
        near = (strength.index >= t - pd.Timedelta(weeks=window)) & (
            strength.index <= t + pd.Timedelta(weeks=window)
        )
        strength[near] = np.nan
    out = pd.DataFrame(rows, columns=["start", "before", "after"])
    out["change"] = out["after"] / out["before"].where(out["before"] > 0) - 1

    acc = by_account.pivot_table(
        index="period", columns="location", values="qty", aggfunc="sum", fill_value=0
    )
    accounts, shares = [], []
    for t in out["start"]:
        pre = acc[(acc.index < t) & (acc.index >= t - pd.Timedelta(weeks=window))].mean()
        post = acc[(acc.index >= t) & (acc.index < t + pd.Timedelta(weeks=window))].mean()
        delta = (post - pre).fillna(0)
        top = delta.abs().idxmax() if len(delta) else None
        accounts.append(top)
        shares.append(
            float(delta[top] / delta.sum()) if top is not None and delta.sum() else np.nan
        )
    out["account"], out["account_share"] = accounts, shares
    return out.sort_values("start").reset_index(drop=True)


# ------------------------------------------------------------------ promos


def promo_check(
    panel: pd.DataFrame,
    expected: pd.Series,
    promos: pd.DataFrame,
    freq: str,
    shift: int,
    flat_below: float = 1.1,
    min_base: float = 20,
) -> pd.DataFrame:
    """Lift per promo day for every recorded promo, moved by `shift`. Measuring per promo
    day keeps a promo that covers only part of a week from looking flat. A promo only
    counts as flat if at least `min_base` units were expected on its promo days; below
    that, noise can't be told apart from a missing uplift."""
    cols = ["promo_start", "promo_end", "location", "qty", "baseline", "lift", "flat"]
    if promos.empty:
        return pd.DataFrame(columns=cols)
    step = pd.Timedelta(days=1 if freq == "D" else 7)
    period_days = step.days
    p = panel.assign(baseline=expected.fillna(0)).set_index("period")
    rows = []
    keys = [promos["promo_start"], promos["promo_end"], promos["location"].fillna("all")]
    for (start, end, loc), _ in promos.groupby(keys):
        days = an.to_period(pd.Series(pd.date_range(start, end, freq="D")), freq)
        promo_days = days.value_counts()
        promo_days.index = promo_days.index + shift * step
        hit = p.reindex(promo_days.index).dropna(subset=["baseline"])
        if hit.empty:
            continue
        on_promo = promo_days.reindex(hit.index) / period_days
        qty, base = hit["qty"].sum(), hit["baseline"].sum()
        base_on_promo = (hit["baseline"] * on_promo).sum()
        lift = 1 + (qty - base) / base_on_promo if base_on_promo > 0 else np.nan
        flat = bool(base_on_promo >= min_base and lift < flat_below)
        rows.append((start, end, loc, qty, base, lift, flat))
    return pd.DataFrame(rows, columns=cols).sort_values("promo_start").reset_index(drop=True)


# -------------------------------------------------------------------- accounts


def accounts(
    sales: pd.DataFrame, end: pd.Timestamp, min_expected: float = 10
) -> pd.DataFrame:
    """Per account (plus 'all' for the whole article): volume, first and last sale,
    and whether it stopped or is new. Stopped = the silence since the last sale should
    have sold `min_expected` units at the account's usual daily rate."""
    sold = sales[sales["qty"] != 0]
    start = sales["date"].min()
    acc = pd.concat(
        [
            sold.groupby("location").agg(
                volume=("qty", "sum"), first_sale=("date", "min"), last_sale=("date", "max")
            ),
            pd.DataFrame(
                {
                    "volume": [sold["qty"].sum()],
                    "first_sale": [sold["date"].min()],
                    "last_sale": [sold["date"].max()],
                },
                index=["all"],
            ),
        ]
    ).rename_axis("location")
    rate = acc["volume"] / ((acc["last_sale"] - acc["first_sale"]).dt.days + 1)
    acc["days_silent"] = (end - acc["last_sale"]).dt.days
    acc["missed"] = rate * acc["days_silent"]
    stopped = (acc["missed"] >= min_expected) & (acc["days_silent"] >= 2)
    new = acc["first_sale"] > start + pd.Timedelta(days=28)
    acc["status"] = np.select(
        [stopped, new],
        [
            "stopped " + acc["last_sale"].dt.strftime("%d %b %Y"),
            "new since " + acc["first_sale"].dt.strftime("%d %b %Y"),
        ],
        "active",
    )
    total = acc.loc["all", "volume"]
    acc["volume_share"] = acc["volume"] / max(total, 1e-9)
    return acc.reset_index().sort_values("volume", ascending=False).reset_index(drop=True)


# -------------------------------------------------------------------- main


def analyse(
    sales: pd.DataFrame,
    promos: pd.DataFrame,
    s: an.Settings,
    holiday_dates: pd.Series | None = None,
    end: pd.Timestamp | None = None,
) -> dict:
    """sales / promos for ONE article (any number of accounts). `end` is the last day
    the data should cover (default: the newest sales day), used to spot stopped sales."""
    s = an.Settings(**{**s.__dict__, "by_location": False})
    r = an.analyse(sales, promos, s)
    panel = r.panel
    row = r.metrics.iloc[0]
    # with fewer than min_events promos the shift isn't trusted; a single article has no
    # portfolio to fall back on, so assume the recorded dates are right
    shift = int(row["best_lag"]) if pd.notna(row["best_lag"]) else 0
    promo = _promo_mask(panel, shift)
    exp = expected_level(panel, s.freq, s.baseline_window, promo, holiday_dates)

    holes = gaps(panel, exp)
    weekly = an.build_panel(sales, promos, an.Settings(freq="W-MON"))
    by_acc = an.build_panel(sales, promos, an.Settings(freq="W-MON", by_location=True))
    in_gap = pd.Series(False, index=weekly["period"])
    for a, b in zip(holes["start"], holes["end"]):
        in_gap |= (in_gap.index >= an.to_period(pd.Series([a]), "W-MON")[0]) & (
            in_gap.index <= b
        )
    shifts = level_shifts(weekly, by_acc, exclude=in_gap)
    checks = promo_check(panel, exp, promos, s.freq, shift)
    end = end if end is not None else sales["date"].max()
    acc = accounts(sales, end)
    stopped = acc[acc["status"].str.startswith("stopped")]
    if "all" in set(stopped["location"]):
        # the whole article stopped: report that once, not again for every account
        # that went quiet on the same day
        last = stopped.loc[stopped["location"] == "all", "last_sale"].iloc[0]
        stopped = stopped[(stopped["location"] == "all") | (stopped["last_sale"] != last)]

    events = pd.concat(
        [
            spikes_and_drops(panel, exp, promo),
            holes,
            pd.DataFrame(
                {
                    "kind": "stopped",
                    "start": stopped["last_sale"] + pd.Timedelta(days=1),
                    "end": end,
                    "account": stopped["location"],
                    "qty": 0.0,
                    "expected": stopped["missed"],
                }
            ),
            pd.DataFrame(
                {
                    "kind": "level shift",
                    "start": shifts["start"],
                    "end": shifts["start"],
                    "account": shifts["account"],
                    "qty": shifts["after"],
                    "expected": shifts["before"],
                }
            ),
            pd.DataFrame(
                {
                    "kind": "flat promo",
                    "start": checks.loc[checks["flat"], "promo_start"],
                    "end": checks.loc[checks["flat"], "promo_end"],
                    "account": checks.loc[checks["flat"], "location"],
                    "qty": checks.loc[checks["flat"], "qty"],
                    "expected": checks.loc[checks["flat"], "baseline"],
                }
            ),
        ],
        ignore_index=True,
    )
    events["change"] = events["qty"] / events["expected"].where(events["expected"] > 0) - 1
    events = events.sort_values(["start", "kind"]).reset_index(drop=True)

    return {
        "panel": panel.assign(expected=exp, promo_effective=promo),
        "weekly": weekly,
        "by_account": by_acc,
        "events": events,
        "level_shifts": shifts,
        "promos": checks,
        "accounts": acc,
        "shift": shift,
        "metrics": row,
    }
