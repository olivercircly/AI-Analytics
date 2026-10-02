"""Shared by all tabs: colours, number formatting and the Context passed to render()."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import analysis as an

WINDOWS = {
    "primary": "Primary (validFrom–validTo)",
    "secondary": "Secondary (retail promo period)",
}
RED, ORANGE, BLUE, GREY, INK = "#c0392b", "#e67e22", "#2980b9", "#95a5a6", "#2c3e50"


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def pct(x: float, signed: bool = False) -> str:
    return "–" if x is None or pd.isna(x) else (f"{x:+.0%}" if signed else f"{x:.0%}")


@dataclass(frozen=True)
class Context:
    """Sidebar settings plus the precomputed results every tab renders from."""

    src: tuple
    cid: str
    demo: bool
    country: str
    promo_window: str
    info: dict
    # granularity
    freq: str
    unit: str
    horizon: int
    test_periods: int
    default_h: int
    default_test: int
    # promotion settings
    by_location: bool
    max_lag: int
    min_events: int
    low: float
    high: float
    # results
    S: dict  # seasonality
    B: dict  # benchmarks
    P: an.Result  # promotions
    H: pd.DataFrame  # holiday effects, whole portfolio
    Q: dict  # data quality

    @property
    def shifted(self) -> pd.DataFrame:
        """Articles whose sales peak away from the recorded promo dates."""
        m = self.P.metrics
        return m[m["best_lag"].fillna(0) != 0]
