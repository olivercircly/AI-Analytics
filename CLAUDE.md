# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit dashboard ("Portfolio analytics") that profiles a forecasting customer's sales history: data quality/latency, seasonality, holiday effects, forecastability benchmarks and promotion effects. Data comes from the `circly_data_handling` MySQL schema, filtered by customer ID (`cid`, a UUID).

## Commands

```bash
source venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest                   # ~2 s, no DB needed
```

Tests in `tests/` run against `demo.make_demo_data()` and assert that each analysis recovers the demo's planted ground truth (shift, outage, lost/half-loaded days, spike, silent account, holiday closures). If you change the demo generator or a detection threshold, update the matching test. There is no linter config or build step. To check the UI without database access, toggle **Use demo data** in the sidebar, or call the pure modules directly:

```bash
python -c "from demo import make_demo_data; import analysis as an; s, p = make_demo_data(); print(an.analyse(s, p, an.Settings()).portfolio)"
```

## Database access

- `db.py` builds the SQLAlchemy engine from `.env` (`DB_SERVER`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_DATABASE`) and requires mutual-TLS certs in `ssl/` (`ca-cert.pem`, `client-cert.pem`, `client-key.pem`). Both `.env` and `ssl/` are gitignored. `.env` holds commented-out blocks for several environments (prod/develop/testing/staging); the active `DB_PORT` picks the one in use, so check which is active before running queries.
- All SQL lives in `queries.py`. Don't use a literal `%` in SQL there, because it clashes with the mysql-connector parameter style.
- DB timestamps are UTC for local midnight (`TIMESTAMPS_ARE_UTC`, `LOCAL_TZ = "Europe/Vienna"`). `_to_local_date` must convert them before bucketing, or every sale moves back one day and looks like a fake 1-day promo shift.
- Promotions have two date windows: **primary** (`validFrom/validTo`, which is the ordering period for producer customers and the promo period for retail customers) and **secondary** (the retailer's shelf promo, filled only for producer customers). `select_window()` maps the chosen one onto `promo_start`/`promo_end`. `promotions.accountId IS NULL` means the promo applies to all accounts.
- `label_articles()` replaces internal `articleId` with `articleNumber` for display, adding the id suffix when numbers collide.

## Architecture

**Layering.** The Streamlit UI is `app.py` (sidebar, data loading, tab layout) plus `views/`: one module per tab with a `render(ctx: Context)` function, shared formatting and colours in `views/common.py`, and the cached loaders and results in `views/data.py`. Analysis lives in pure pandas modules with no Streamlit or DB imports (`analysis.py`, `seasonality.py`, `calendar_effects.py`, `data_quality.py`, `benchmark.py`), so they can be tested on their own. `queries.py` uses Streamlit only for `st.cache_data`/`st.cache_resource`. `views/data.py` imports `queries` and `demo` lazily so that demo mode works without DB credentials or certs. Keep that separation.

**Data contract.** Every analysis takes the same two frames:
- sales: `date, product, location, qty` (DB data also has `article_number`)
- promos: `product, location, promo_start, promo_end` (`location` None = all accounts)

`demo.make_demo_data()` produces the same shape, with known ground truth built in: a -7 day distributor shift on half the products, an outage, lost and half-loaded days, a data-entry spike, and an account going silent. `make_demo_quality()` mimics `QUALITY_SQL` output. If you add a column to the contract, add it to the demo generator too.

**The panel.** `analysis.build_panel()` turns sales and promos into a complete per-series × period grid. Missing periods inside a series' active range become zero, and partial weeks at both ends are dropped (a week counts as complete once it reaches Saturday). It adds `promo_days`, `promo` and `series` columns. `freq` is `"D"` or `"W-MON"`. Seasonality, holidays, data quality and benchmarks all reuse this panel. `analysis.group_rolling()` is the shared vectorised per-group rolling helper; use it instead of `groupby().transform(lambda ... rolling)`, which is much slower.

**Caching and context.** `src = (cid, start, promo_window, demo)` identifies the dataset, and every cached function (`load`, `panel`, `promo_result`, `season_result`, `holiday_result`, `dq_result`, `bench_result`) is keyed on it plus its own parameters. All results are computed up front in `app.py`, before the tabs render, and passed to each view in a `views.common.Context` together with the sidebar settings. Add a new setting or result to `Context` rather than recomputing it inside a view. Settings are collected in a sidebar `st.form`, so nothing reruns until **Run analysis** is pressed. `cid` is mirrored to `st.query_params` so URLs can be shared. `GRANULARITY` in `app.py` sets the per-frequency defaults (baseline window, max lag, benchmark horizon, test periods).

**Domain conventions** (see module docstrings for details):
- Promo lag = sales period − recorded promo period. Negative means sales come before the recorded promo (e.g. distributor sell-in). `lag_scores` tests shifts in both directions, and the pooled best shift becomes the portfolio `shift`.
- The promo baseline is a centred rolling median over non-promo periods. Dependency bands (`low_dependency`/`high_dependency`) classify the incremental share.
- Seasonality uses a multiplicative decomposition with Hyndman-style strength and excludes promo periods.
- Holiday baseline is the median of the same weekday 1–4 weeks before and after, skipping days near holidays. `calendar_effects.COUNTRIES` lists the supported calendars.
- Data quality judges missing days per account against that account's normal week, excluding holidays.
- Benchmarks backtest simple methods with WAPE and bias. A production model should beat the best of them.
