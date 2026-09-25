"""Holiday effects: how sales change in the days around each public holiday.

Baseline for a day = median of the same weekday 1-4 weeks before and after, skipping
days near any holiday. So "-100% on Christmas Day" means closed, "+40% the day before"
means pre-holiday shopping.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd
from dateutil.easter import easter

COUNTRIES = {
    "AT": "Austria",
    "DE": "Germany",
    "CH": "Switzerland",
    "IT": "Italy",
    "SI": "Slovenia",
    "CZ": "Czechia",
    "HU": "Hungary",
    "SK": "Slovakia",
}


def holiday_calendar(
    country: str, years: list[int], subdiv: str | None = None
) -> pd.DataFrame:
    try:
        public = holidays.country_holidays(
            country, subdiv=subdiv, years=years, language="en_US"
        )
    except (NotImplementedError, KeyError, ValueError):
        public = holidays.country_holidays(country, subdiv=subdiv, years=years)
    rows = [(pd.Timestamp(d), n.split(";")[0].strip(), True) for d, n in public.items()]

    # not public holidays, but they drive retail demand
    for y in years:
        e = pd.Timestamp(easter(y))
        rows += [
            (e, "Easter Sunday", False),
            (e + pd.Timedelta(days=49), "Pentecost Sunday", False),
            (pd.Timestamp(y, 12, 24), "Christmas Eve", False),
            (pd.Timestamp(y, 12, 31), "New Year's Eve", False),
        ]
    cal = pd.DataFrame(rows, columns=["date", "name", "public"])

    # bridge days: Monday before a Tuesday holiday, Friday after a Thursday holiday
    pub = cal[cal["public"]]
    bridges = pd.concat(
        [
            pub.loc[pub["date"].dt.dayofweek == 1, "date"] - pd.Timedelta(days=1),
            pub.loc[pub["date"].dt.dayofweek == 3, "date"] + pd.Timedelta(days=1),
        ]
    )
    bridges = bridges[~bridges.isin(cal["date"])]
    cal = pd.concat(
        [cal, pd.DataFrame({"date": bridges, "name": "Bridge day", "public": False})]
    )
    return (
        cal.drop_duplicates(["date", "name"]).sort_values("date").reset_index(drop=True)
    )


def relative_to_baseline(
    daily_total: pd.Series, calendar: pd.DataFrame, guard: int = 3
) -> pd.Series:
    """daily_total: index = date, values = units. Returns sales / weekday baseline - 1."""
    y = daily_total.asfreq("D", fill_value=0.0)
    near = pd.Series(False, index=y.index)
    for off in range(-guard, guard + 1):
        near |= y.index.isin(calendar["date"] + pd.Timedelta(days=off))
    clean = y.where(~near)
    base = pd.concat(
        [clean.shift(7 * k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)], axis=1
    ).median(axis=1)
    return (y / base.where(base > 0)) - 1


def holiday_effects(
    daily_total: pd.Series, country: str, window: int = 3, subdiv: str | None = None
) -> pd.DataFrame:
    """Long table: holiday, offset (days), effect (relative), occurrences."""
    if daily_total.empty:
        return pd.DataFrame(
            columns=["holiday", "offset", "effect", "occurrences", "order"]
        )
    lo, hi = daily_total.index.min(), daily_total.index.max()
    cal = holiday_calendar(country, list(range(lo.year, hi.year + 1)), subdiv)
    rel = relative_to_baseline(daily_total, cal)
    # only occurrences with a full window of data and a usable baseline around them
    cal = cal[
        (cal["date"] - pd.Timedelta(days=35) >= lo)
        & (cal["date"] + pd.Timedelta(days=35) <= hi)
    ]

    rows = []
    for name, grp in cal.groupby("name"):
        order = grp["date"].dt.dayofyear.median()
        for off in range(-window, window + 1):
            vals = rel.reindex(grp["date"] + pd.Timedelta(days=off)).dropna()
            vals = vals[np.isfinite(vals)]
            rows.append(
                (name, off, vals.mean() if len(vals) else np.nan, len(vals), order)
            )
    return pd.DataFrame(
        rows, columns=["holiday", "offset", "effect", "occurrences", "order"]
    )


def summarise(effects: pd.DataFrame) -> dict:
    if effects.empty or effects["effect"].isna().all():
        return {"closed": [], "biggest_uplift": None, "biggest_drop": None}
    on_day = effects[effects["offset"] == 0].set_index("holiday")["effect"]
    near = effects[effects["offset"] != 0].dropna(subset=["effect"])
    up = near.loc[near["effect"].idxmax()] if len(near) else None
    down = on_day[(on_day > -0.9) & (on_day < -0.1)]
    return {
        "closed": sorted(on_day[on_day <= -0.9].index),
        "biggest_uplift": (
            (up["holiday"], int(up["offset"]), float(up["effect"]))
            if up is not None
            else None
        ),
        "biggest_drop": (down.idxmin(), float(down.min())) if len(down) else None,
    }
