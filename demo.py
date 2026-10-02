"""Synthetic sales + promos with known ground truth, for testing without the database."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def make_demo_data(n_products: int = 40, shift_days: int = -7, seed: int = 0):
    """Shops closed on Sundays and Austrian public holidays, busier the day before.
    Each product has its own size, yearly season (summer or winter peak) and promos.
    Half the products are sold through a distributor that buys `shift_days`
    before the store promo starts (negative = earlier)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-02", "2025-12-28", freq="D")
    dow, doy = days.dayofweek.to_numpy(), days.dayofyear.to_numpy()
    closed = days.isin(
        pd.to_datetime(list(holidays.country_holidays("AT", years=[2023, 2024, 2025])))
    )
    closed |= dow == 6
    before_closed = np.roll(closed, -1) & ~closed

    # data problems: one system-wide outage, two lost days and two half-loaded days per
    # account, and account 1406 stops sending data 20 days before the end
    accounts = ["1404", "1405", "1406"]
    open_days = np.flatnonzero(~closed)
    outage = rng.choice(open_days[100:-60], 1)
    lost = {a: rng.choice(open_days[100:-60], 2) for a in accounts}
    half = {a: rng.choice(open_days[100:-60], 2) for a in accounts}

    sales, promos = [], []
    for i in range(n_products):
        product = f"{9700 + i:06d}"
        account = accounts[i % 3]
        base = float(np.exp(rng.uniform(np.log(0.5), np.log(60))))
        weekly = 1 + 0.3 * np.sin(2 * np.pi * dow / 7)
        yearly = 1 + rng.uniform(0, 0.6) * np.cos(
            2 * np.pi * (doy - rng.choice([15, 196])) / 365.25
        )
        demand = base * weekly * yearly
        demand[before_closed] *= 1.4
        n_promos = rng.integers(0, 14)
        lift = rng.uniform(0.3, 2.5)
        shifted = i % 2 == 0
        for start in rng.choice(len(days) - 40, size=n_promos, replace=False):
            start_day, length = days[start], int(rng.integers(7, 15))
            promos.append(
                (product, None, start_day, start_day + pd.Timedelta(days=length - 1))
            )
            s = start + (shift_days if shifted else 0)
            demand[s : s + length] *= 1 + lift
            demand[s + length : s + length + 7] *= 0.8  # post-promo dip
        demand[closed] = 0
        qty = rng.poisson(demand)
        qty[np.r_[outage, lost[account]]] = 0
        if i % 2:
            qty[half[account]] = 0
        if i == 5:
            qty[500] *= 25  # data-entry error
        if account == "1406":
            qty[-20:] = 0
        sales.append(
            pd.DataFrame(
                {"date": days, "product": product, "location": account, "qty": qty}
            )
        )
    return (
        pd.concat(sales, ignore_index=True),
        pd.DataFrame(
            promos, columns=["product", "location", "promo_start", "promo_end"]
        ),
    )


def make_demo_quality(
    sales: pd.DataFrame, onboarded: str = "2025-01-06", seed: int = 0
) -> pd.DataFrame:
    """Arrival statistics like QUALITY_SQL returns: history before onboarding was loaded in one
    backfill, afterwards most rows arrive the next day, some take up to two weeks."""
    rng = np.random.default_rng(seed)
    rows = sales[sales["qty"] != 0].groupby("date").size()
    onboard = pd.Timestamp(onboarded)
    out = []
    lags = np.array([6, 12, 24, 30, 48, 72, 96, 120, 168, 240, 336])
    p = np.array([0.05, 0.30, 0.30, 0.10, 0.10, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01])
    for day, n in rows.items():
        if day < onboard:
            out.append((day, float((onboard - day).days * 24), n))
        else:
            for lag, k in zip(lags, rng.multinomial(n, p)):
                if k:
                    out.append((day, float(lag), k))
    q = pd.DataFrame(out, columns=["date", "lag_hours", "n_rows"])
    q["negative_rows"] = rng.binomial(q["n_rows"], 0.003)
    q["zero_rows"] = rng.binomial(q["n_rows"], 0.001)
    q["updated_rows"] = rng.binomial(
        q["n_rows"], np.where(q["date"] > pd.Timestamp("2025-06-01"), 0.02, 0.002)
    )
    q["deleted_rows"] = rng.binomial(q["n_rows"], 0.004)
    return q
