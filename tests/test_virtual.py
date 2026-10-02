"""Rolling stores up into virtual accounts (accounts.isVirtual)."""

import pandas as pd

import analysis as an
import data_quality as dq
from demo import make_demo_virtual

V1400 = "1400 (virtual V-1400)"


def test_virtual_groups_labels_and_lowest_id_wins():
    mapping = pd.DataFrame(
        {
            "v_accountId": ["20", "10", "10"],
            "v_externalAccountId": ["V-B", "V-A", "V-A"],
            "r_accountId": ["1", "1", "2"],
            "r_externalAccountId": ["S-1", "S-1", "S-2"],
        }
    )
    assert an.virtual_groups(mapping) == {"1": "10 (virtual V-A)", "2": "10 (virtual V-A)"}
    assert an.virtual_groups(mapping.iloc[:0]) == {}


def test_roll_up_moves_sales_and_store_promos():
    sales = pd.DataFrame(
        {"date": pd.Timestamp("2024-01-01"), "product": "A", "location": ["1", "2", "3"], "qty": 1.0}
    )
    promos = pd.DataFrame(
        {
            "product": "A",
            "location": [None, "1", "3"],
            "promo_start": pd.Timestamp("2024-01-01"),
            "promo_end": pd.Timestamp("2024-01-07"),
        }
    )
    s, p = an.roll_up_accounts(sales, promos, {"1": "V", "2": "V"})
    assert s["location"].tolist() == ["V", "V", "3"]
    assert pd.isna(p["location"].iloc[0])  # missing = all accounts, unchanged
    assert p["location"].iloc[1:].tolist() == ["V", "3"]


def test_demo_rollup_keeps_volume_and_merges_accounts(sales, promos, holiday_dates, today):
    s, p = an.roll_up_accounts(sales, promos, an.virtual_groups(make_demo_virtual()))
    assert s["qty"].sum() == sales["qty"].sum()
    assert set(s["location"]) == {V1400, "1406"}

    daily = an.build_panel(s, p, an.Settings(freq="D"))
    acc = dq.analyse(s, daily, None, holiday_dates, today)["accounts"].set_index("location")
    assert acc.loc[V1400, "status"] == "active"
    assert acc.loc["1406", "status"].startswith("silent")
    assert acc.loc[V1400, "volume"] == sales.loc[sales["location"].isin(["1404", "1405"]), "qty"].sum()


def test_split_by_account_uses_virtual_accounts(sales, promos):
    s, p = an.roll_up_accounts(sales, promos, an.virtual_groups(make_demo_virtual()))
    r = an.analyse(s, p, an.Settings(by_location=True))
    assert set(r.metrics["location"]) == {V1400, "1406"}


def test_article_overlap_counts_shared_and_exclusive_articles():
    rows = [  # (product, location, qty)
        ("A", "V1", 5), ("A", "V2", 1),  # shared by V1 and V2
        ("B", "V1", 2),                  # only V1
        ("C", "V2", 3), ("C", "9", 4),   # only V2, but also at an outside store
        ("D", "9", 1),                   # outside only
        ("E", "V3", 0),                  # no positive sales: not sold
    ]
    sales = pd.DataFrame(rows, columns=["product", "location", "qty"])
    o = an.article_overlap(sales, ["V1", "V2", "V3"])

    art = o["articles"].set_index("product")
    assert art["groups"].to_dict() == {"A": 2, "B": 1, "C": 1, "D": 0}
    assert art.loc["A", "where"] == "V1, V2"
    assert art["outside"].to_dict() == {"A": False, "B": False, "C": True, "D": True}

    acc = o["accounts"].set_index("location")
    assert set(acc.index) == {"V1", "V2"}
    assert acc.loc["V1", ["articles", "exclusive", "shared"]].tolist() == [2, 1, 1]
    assert acc.loc["V1", "shared_volume_share"] == 5 / 7

    pairs = o["pairs"].set_index(["a", "b"])
    assert pairs.loc[("V1", "V2"), "shared"] == 1
    assert pairs.loc[("V1", "V2"), "jaccard"] == 1 / 3
    assert pairs.loc[("V2", an.OUTSIDE), "shared"] == 1
    assert ("V1", an.OUTSIDE) not in pairs.index


def test_demo_virtual_account_is_disjoint_from_the_other_store(sales, promos):
    s, _ = an.roll_up_accounts(sales, promos, an.virtual_groups(make_demo_virtual()))
    o = an.article_overlap(s, [V1400])
    assert o["accounts"]["shared"].tolist() == [0]
    art = o["articles"]
    assert not art.loc[art["groups"] > 0, "outside"].any()  # 1406 sells other articles
    assert art["outside"].any()
