import numpy as np
import pandas as pd

import analysis as an

NO_PROMOS = pd.DataFrame(columns=["product", "location", "promo_start", "promo_end"])


def _sales(dates, product="A", location="1", qty=1.0):
    return pd.DataFrame(
        {"date": pd.to_datetime(dates), "product": product, "location": location, "qty": qty}
    )


def test_group_rolling_matches_transform():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 50), "x": rng.normal(size=150)})
    df = df.sample(frac=1, random_state=0)  # groups interleaved, not sorted
    fast = an.group_rolling(df["x"], df["g"], 7, "median", center=True, min_periods=3)
    slow = df.groupby("g")["x"].transform(
        lambda x: x.rolling(7, center=True, min_periods=3).median()
    )
    pd.testing.assert_series_equal(fast, slow, check_names=False)


def test_daily_panel_fills_gaps_inside_active_range_only():
    sales = pd.concat(
        [_sales(["2024-01-01", "2024-01-04"], "A"), _sales(["2024-01-02", "2024-01-06"], "B")]
    )
    p = an.build_panel(sales, NO_PROMOS, an.Settings(freq="D"))
    a = p[p["product"] == "A"]
    assert a["period"].tolist() == list(pd.date_range("2024-01-01", "2024-01-04"))
    assert a["qty"].tolist() == [1, 0, 0, 1]
    assert p.loc[p["product"] == "B", "period"].min() == pd.Timestamp("2024-01-02")


def test_weekly_panel_drops_partial_weeks():
    # starts on a Wednesday, ends on a Friday: first and last week are incomplete
    sales = _sales(pd.date_range("2024-01-03", "2024-01-26"))
    p = an.build_panel(sales, NO_PROMOS, an.Settings(freq="W-MON"))
    assert p["period"].tolist() == [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")]
    assert (p["qty"] == 7).all()


def test_weekly_panel_keeps_week_ending_on_saturday():
    # shops closed on Sunday: a week counts as complete once it reaches Saturday
    sales = _sales(pd.date_range("2024-01-01", "2024-01-13"))
    p = an.build_panel(sales, NO_PROMOS, an.Settings(freq="W-MON"))
    assert p["period"].max() == pd.Timestamp("2024-01-08")


def test_promo_without_location_applies_to_all_accounts():
    days = pd.date_range("2024-01-01", "2024-01-10")
    sales = pd.concat([_sales(days, location="1"), _sales(days, location="2")])
    promos = pd.DataFrame(
        {
            "product": ["A", "A"],
            "location": [None, "2"],
            "promo_start": pd.to_datetime(["2024-01-03", "2024-01-08"]),
            "promo_end": pd.to_datetime(["2024-01-04", "2024-01-08"]),
        }
    )
    p = an.build_panel(sales, promos, an.Settings(freq="D", by_location=True))
    on = p[p["promo"]].groupby("location")["period"].apply(list).to_dict()
    assert on["1"] == list(pd.to_datetime(["2024-01-03", "2024-01-04"]))
    assert on["2"] == list(pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-08"]))


def test_promo_days_counted_per_week():
    sales = _sales(pd.date_range("2024-01-01", "2024-01-20"))
    promos = pd.DataFrame(
        {
            "product": ["A"],
            "location": [None],
            "promo_start": [pd.Timestamp("2024-01-05")],
            "promo_end": [pd.Timestamp("2024-01-10")],
        }
    )
    p = an.build_panel(sales, promos, an.Settings(freq="W-MON")).set_index("period")
    assert p["promo_days"].to_dict() == {
        pd.Timestamp("2024-01-01"): 3,
        pd.Timestamp("2024-01-08"): 3,
        pd.Timestamp("2024-01-15"): 0,
    }


def test_less_than_a_week_gives_an_empty_weekly_panel(sales, promos):
    short = sales[sales["date"] >= sales["date"].max() - pd.Timedelta(days=2)]
    p = an.build_panel(short, promos, an.Settings(freq="W-MON"))
    assert p.empty
    assert {"period", "qty", "promo_days", "promo", "series"} <= set(p.columns)
