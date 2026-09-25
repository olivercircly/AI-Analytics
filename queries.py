"""Database access. Adjust the two SQL statements to your schema; everything else adapts."""

from __future__ import annotations

from datetime import date

import pandas as pd
import sqlalchemy as sa
import streamlit as st

from db import create_engine

SCHEMA = "circly_data_handling"

# Your timestamps look like UTC for local midnight (2021-11-21 23:00:00 = 22 Nov in Vienna).
# Bucketing them as-is would move every sale one day back and fake a 1-day "promo shift".
TIMESTAMPS_ARE_UTC = True
LOCAL_TZ = "Europe/Vienna"

# sales.accountId = the ordering account (store / distributor), promotions.accountId NULL = all accounts.
# Rows are pre-aggregated per timestamp x article x account to keep the transfer small
# (uses index IDX_cid_orderDate_article).
SALES_SQL = sa.text(f"""
    SELECT
        s.orderDate                   AS ts,
        CAST(s.articleId AS CHAR)     AS product,
        MIN(s.articleNumber)          AS article_number,
        CAST(s.accountId AS CHAR)     AS location,
        SUM(s.quantity)               AS qty
    FROM {SCHEMA}.sales s
    WHERE s.cid = :cid
      AND s.orderDate >= :start
      AND s.type = 'sales'
      AND s.deleted_at IS NULL
    GROUP BY s.orderDate, s.articleId, s.accountId
""")

# Primary window: ordering period (producer customers) or promo period (retail customers).
# Secondary window: the retailer's shelf promo period, only filled for producer customers.
PROMO_SQL = sa.text(f"""
    SELECT
        CAST(p.articleId AS CHAR)     AS product,
        CAST(p.accountId AS CHAR)     AS location,
        p.validFrom                   AS primary_start,
        p.validTo                     AS primary_end,
        p.validFromSecondary          AS secondary_start,
        p.validToSecondary            AS secondary_end,
        p.externalPromotionID         AS promo_id,
        p.type, p.strength, p.placement, p.region, p.targetGroup
    FROM {SCHEMA}.promotions p
    WHERE p.cid = :cid
      AND p.validTo >= :start
      AND p.deleted_at IS NULL
""")

PROMO_WINDOWS = {
    "primary": "Primary (validFrom–validTo)",
    "secondary": "Secondary (retail promo period)",
}


@st.cache_resource
def get_engine() -> sa.Engine:
    return create_engine()


def _to_local_date(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    if TIMESTAMPS_ARE_UTC:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return ts.dt.normalize()


def _read(sql: sa.TextClause, **params) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params=params)


@st.cache_data(ttl=3600, show_spinner="Loading sales…")
def load_sales(cid: str, start: date) -> pd.DataFrame:
    df = _read(SALES_SQL, cid=cid, start=start)
    df["date"] = _to_local_date(df["ts"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["product"] = df["product"].astype(str)
    df["article_number"] = df["article_number"].astype(str).str.strip()
    df["location"] = df["location"].fillna("all").astype(str)
    return df[["date", "product", "article_number", "location", "qty"]]


@st.cache_data(ttl=3600, show_spinner="Loading promotions…")
def load_promos(cid: str, start: date) -> pd.DataFrame:
    """All promotion rows with both windows; pick one with select_window()."""
    df = _read(PROMO_SQL, cid=cid, start=start)
    for col in ["primary_start", "primary_end", "secondary_start", "secondary_end"]:
        df[col] = _to_local_date(df[col])
    df["product"] = df["product"].astype(str)
    df["location"] = df["location"].where(df["location"].notna(), None)
    return df


def select_window(promos: pd.DataFrame, window: str) -> pd.DataFrame:
    """Map the chosen window onto promo_start / promo_end; rows without it are dropped."""
    out = promos.assign(
        promo_start=promos[f"{window}_start"], promo_end=promos[f"{window}_end"]
    )
    return out.dropna(subset=["promo_start", "promo_end"])


def label_articles(
    sales: pd.DataFrame, promos: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Show articleNumber instead of the internal articleId where that is unambiguous."""
    names = sales.groupby("product")["article_number"].first()
    dupes = names.duplicated(keep=False)
    names[dupes] = names[dupes] + " (#" + names.index[dupes] + ")"
    return (
        sales.assign(product=sales["product"].map(names)),
        promos.assign(product=promos["product"].map(names)).dropna(subset=["product"]),
    )
