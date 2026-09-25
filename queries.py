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

# TODO: replace the placeholder column names (see the "Schema" tab in the app).
# Every query must return exactly these aliases. Keep CAST(... AS CHAR) so IDs
# from sales and promotions compare as equal strings.
SALES_SQL = sa.text(f"""
    SELECT
        s.orderDate                 AS ts,
        CAST(s.articleId  AS CHAR)  AS product,
        SUM(s.quantity)              AS qty
    FROM {SCHEMA}.sales s
    WHERE s.cid = :cid
      AND s.type = 'sales'
      AND s.orderDate >= :start
    GROUP BY 1, 2, 3
""")

PROMO_SQL = sa.text(f"""
    SELECT
        CAST(p.product_id  AS CHAR)  AS product,
        CAST(p.location_id AS CHAR)  AS location,   -- NULL = promo applies to all locations
        p.start_date                 AS promo_start,
        p.end_date                   AS promo_end
    FROM {SCHEMA}.promotions p
    WHERE p.cid = :cid
      AND p.validTo >= :start
""")

SCHEMA_SQL = sa.text("""
    SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name,
           DATA_TYPE AS data_type, IS_NULLABLE AS nullable
    FROM information_schema.columns
    WHERE TABLE_SCHEMA = :schema
    ORDER BY TABLE_NAME, ORDINAL_POSITION
""")


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
    df["location"] = df["location"].fillna("all").astype(str)
    return df[["date", "product", "location", "qty"]]


@st.cache_data(ttl=3600, show_spinner="Loading promotions…")
def load_promos(cid: str, start: date) -> pd.DataFrame:
    df = _read(PROMO_SQL, cid=cid, start=start)
    df["promo_start"] = _to_local_date(df["promo_start"])
    df["promo_end"] = _to_local_date(df["promo_end"])
    df["product"] = df["product"].astype(str)
    df["location"] = df["location"].where(df["location"].notna(), None)
    return df[["product", "location", "promo_start", "promo_end"]]


@st.cache_data(ttl=86400, show_spinner="Reading schema…")
def describe_schema() -> pd.DataFrame:
    return _read(SCHEMA_SQL, schema=SCHEMA)
