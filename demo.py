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

    sales, promos = [], []
    for i in range(n_products):
        product = f"{9700 + i:06d}"
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
        sales.append(
            pd.DataFrame(
                {"date": days, "product": product, "location": "all", "qty": qty}
            )
        )
    return (
        pd.concat(sales, ignore_index=True),
        pd.DataFrame(
            promos, columns=["product", "location", "promo_start", "promo_end"]
        ),
    )
