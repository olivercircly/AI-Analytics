"""DB-free parts of queries.py (importing it needs no credentials)."""

import pandas as pd
import pytest

queries = pytest.importorskip("queries")


def test_utc_midnight_becomes_local_day():
    ts = pd.Series(["2021-11-21 23:00:00", "2024-06-30 22:00:00"])  # CET and CEST
    got = queries._to_local_date(ts)
    assert got.tolist() == [pd.Timestamp("2021-11-22"), pd.Timestamp("2024-07-01")]


def test_select_window_drops_rows_without_that_window():
    promos = pd.DataFrame(
        {
            "product": ["A", "B"],
            "primary_start": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "primary_end": pd.to_datetime(["2024-01-07", "2024-02-07"]),
            "secondary_start": pd.to_datetime(["2024-01-08", None]),
            "secondary_end": pd.to_datetime(["2024-01-14", None]),
        }
    )
    sec = queries.select_window(promos, "secondary")
    assert sec["product"].tolist() == ["A"]
    assert sec["promo_start"].iloc[0] == pd.Timestamp("2024-01-08")
    assert len(queries.select_window(promos, "primary")) == 2


def test_label_articles_disambiguates_duplicate_numbers():
    sales = pd.DataFrame(
        {"product": ["1", "2", "3"], "article_number": ["X-1", "X-1", "Y-2"], "qty": 1}
    )
    promos = pd.DataFrame({"product": ["3", "99"]})
    s, p = queries.label_articles(sales, promos)
    assert s["product"].tolist() == ["X-1 (#1)", "X-1 (#2)", "Y-2"]
    assert p["product"].tolist() == ["Y-2"]  # promos for unknown articles are dropped
