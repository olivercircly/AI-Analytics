"""Shared fixtures: the demo data set (with its known ground truth) and panels built from it."""

import pandas as pd
import pytest

import analysis as an
import calendar_effects as ce
from demo import make_demo_data, make_demo_quality


def is_shifted(product: str) -> bool:
    """demo.make_demo_data shifts every even-numbered product."""
    return (int(product) - 9700) % 2 == 0


@pytest.fixture(scope="session")
def demo():
    return make_demo_data()


@pytest.fixture(scope="session")
def sales(demo):
    return demo[0]


@pytest.fixture(scope="session")
def promos(demo):
    return demo[1]


@pytest.fixture(scope="session")
def daily(sales, promos):
    return an.build_panel(sales, promos, an.Settings(freq="D"))


@pytest.fixture(scope="session")
def weekly(sales, promos):
    return an.build_panel(sales, promos, an.Settings(freq="W-MON"))


@pytest.fixture(scope="session")
def quality(sales):
    return make_demo_quality(sales)


@pytest.fixture(scope="session")
def holiday_dates(sales):
    years = list(range(sales["date"].min().year, sales["date"].max().year + 1))
    return ce.holiday_calendar("AT", years)["date"]


@pytest.fixture(scope="session")
def today(sales):
    return sales["date"].max() + pd.Timedelta(days=1)
