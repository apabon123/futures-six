# VRP Data Requirements Spec

**Purpose:** Minimal data ingestion requirements so the VRP Meta-Sleeve (Core, Convergence, Alt) produces signals on all rebalance dates. No backtests; code pointers and table/symbol mapping only.

---

## 1. VIX/VX-Related Inputs Required by VRP Sleeves

### 1.1 Required series (names/symbols)

| Series | Symbol/ID | Used by | Purpose |
|--------|-----------|---------|---------|
| **VIX (1M)** | FRED `VIXCLS` | VRP Core, Convergence, Alt, policy features | Implied vol level; VRP spread (VIX − RV), convergence spread (VIX − VX1), Alt-VRP (VIX − RV5) |
| **VIX3M** | CBOE `VIX3M` | VRP Core (passthrough in features), load_vrp_inputs | Term structure; combined loader only |
| **VVIX** | CBOE `VVIX` or FRED `VVIXCLS` | VRP stress proxy, gamma_stress (feature_gamma_stress) | VRP gating: OFF when VVIX ≥ 99th percentile; policy stress |
| **VX1** | `@VX=101XN` | All VRP strategies + vx_backwardation | Front-month VX futures; signal instrument; backwardation (VX1 > VX2) |
| **VX2** | `@VX=201XN` | Convergence (curve_slope), vx_backwardation, Crisis sleeve | Second-month VX; backwardation flag; VX2−VX1 spread |
| **VX3** | `@VX=301XN` | Core/Convergence (load_vrp_inputs), VX carry strategies | Third-month VX; optional for curve slope / carry |
| **ES continuous** | `ES_FRONT_CALENDAR_2D` | VRP Core (RV21), VRP Alt (RV5) | Realized vol from MarketData `returns_cont` (not from `load_rv` in production features) |

### 1.2 Where each series is expected

- **MarketData broker** (single OHLCV table, e.g. `g_continuous_bar_daily`):
  - **ES_FRONT_CALENDAR_2D** — Required. VRP Core uses `market.returns_cont[es_symbol]` for 21d realized vol; VRP Alt uses it for 5d realized vol. Column used: symbol/contract_series per table schema; field: `close` (returns derived in broker).
- **VX1/VX2/VX3** — Not read by MarketData. The broker’s universe is configured in `configs/data.yaml` and does not include VX. VX is read only via **vrp_loaders** and direct SQL in diagnostics.

- **VRP loaders** (`src/market_data/vrp_loaders.py`) — all use the **same DuckDB** as MarketData (path from `configs/data.yaml` or `db_path`), but different tables:
  - **VIX:** `f_fred_observations` (series_id = `VIXCLS`); columns: `date`, `value`.
  - **VIX3M:** `market_data_cboe` (symbol = `VIX3M`); columns: `timestamp`, `settle`.
  - **VVIX:** `market_data_cboe` (symbol = `VVIX`) first; fallback `f_fred_observations` (series_id = `VVIXCLS`).
  - **VX1/2/3:** `market_data` (symbols `@VX=101XN`, `@VX=201XN`, `@VX=301XN`); columns: `timestamp`, `close`.

- **Realized vol (ES):** VRP Core and VRP Alt compute RV from **MarketData** `returns_cont` (i.e. from the OHLCV table used by MarketData, e.g. `g_continuous_bar_daily` with `ES_FRONT_CALENDAR_2D`). The function `load_rv()` in vrp_loaders reads ES from table `market_data` and is used in diagnostics/optional paths, not in the main feature compute path for Core/Alt.

---

## 2. Mapping to DuckDB Tables

### 2.1 Table used by MarketData (verify_setup)

- **Table:** Discovered at runtime via `find_ohlcv_table(conn)` in `src/agents/utils_db.py` — typically the table with required OHLCV columns and the most rows (e.g. **`g_continuous_bar_daily`**).
- **Columns (conceptual):** date (or trading_date), symbol (or contract_series), open, high, low, close, volume.
- **Scope:** Universe only (ES, NQ, RTY, ZT, ZF, ZN, UB, SR3, CL, GC, 6E, 6B, 6J). **VX is not in this universe**; VRP does not read VX from this table.

### 2.2 Tables required for VRP (separate from MarketData’s OHLCV table)

| Table | Required for VRP | Columns used | Symbols/IDs |
|-------|------------------|--------------|--------------|
| **f_fred_observations** | Yes | `date`, `series_id`, `value` | `VIXCLS`; optional `VVIXCLS` (VVIX fallback) |
| **market_data_cboe** | Yes | `timestamp`, `symbol`, `settle` | `VIX3M`, `VVIX` |
| **market_data** | Yes | `timestamp`, `symbol`, `close` | `@VX=101XN`, `@VX=201XN`, `@VX=301XN` |

- **ES in market_data:** Optional for VRP feature code (Core/Alt use MarketData’s returns). Used by `load_rv()` and by diagnostics (e.g. VX1 returns from `market_data` for PnL). So for **signals only**, ES can live only in `g_continuous_bar_daily`; for **diagnostics/phase1** that query VX1 returns (and any path using `load_rv` from DB), `market_data` is also used.
- **Same DB:** MarketData and vrp_loaders both use `configs/data.yaml` → `db.path` (or explicit `db_path`). So all of the above tables are expected in the **same** DuckDB database (e.g. under `../databento-es-options/data/silver`).

### 2.3 Summary: Does VRP need more than `g_continuous_bar_daily`?

- **Yes.** VRP expects **three additional** table/source types:
  1. **f_fred_observations** — VIX (VIXCLS), and optionally VVIX (VVIXCLS).
  2. **market_data_cboe** — VIX3M, VVIX.
  3. **market_data** — VX1/2/3 (`@VX=101XN`, `@VX=201XN`, `@VX=301XN`).

Without these, VRP feature modules and policy features (vx_backwardation, vrp_stress_proxy) cannot run. `g_continuous_bar_daily` alone only provides the **ES** returns used for realized vol in VRP Core and VRP Alt; it does not provide VIX, VIX3M, VVIX, or VX curve.

---

## 3. Minimal Data Ingestion Additions (symbols + fields + frequency)

To have VRP produce signals on all rebalance dates:

### 3.1 Symbols / series to ingest

- **FRED:** `VIXCLS` (daily). Optional: `VVIXCLS` if not using CBOE VVIX.
- **CBOE (market_data_cboe):** `VIX3M`, `VVIX` — daily settle.
- **VX futures (market_data):** `@VX=101XN`, `@VX=201XN`, `@VX=301XN` — daily **close** (and timestamp for date). Unadjusted, 1-day roll continuous contracts as in vrp_loaders docstring.

### 3.2 Fields

- **f_fred_observations:** `date`, `series_id`, `value` (as DOUBLE for VIX).
- **market_data_cboe:** `timestamp` (or date), `symbol`, `settle`.
- **market_data:** `timestamp` (or date), `symbol`, `close`.

### 3.3 Frequency

- **Daily** for all. Rebalance dates are daily; no intraday or lower frequency is required for VRP signals.

### 3.4 ES for VRP

- **ES_FRONT_CALENDAR_2D** must be present in the **MarketData OHLCV table** (e.g. `g_continuous_bar_daily`) so that `market.returns_cont` includes ES and VRP Core/Alt can compute RV21/RV5. No extra table is required for ES if it is already in that OHLCV table.

### 3.5 Optional for full diagnostics / load_rv

- **market_data** containing **ES_FRONT_CALENDAR_2D** and **@VX=101XN** for:
  - Phase-1 diagnostics that load VX1 returns from DB (`src/diagnostics/vrp_core_phase1.py`, `vrp_convergence_phase1.py` query `market_data` for VX1 close).
  - `load_rv(..., symbol="ES_FRONT_CALENDAR_2D")` if any caller uses it (e.g. research); not required for the main strategy feature path.

---

## 4. Code Pointers (file paths)

| Concern | File(s) |
|--------|--------|
| VRP loaders (VIX, VIX3M, VVIX, VX curve, load_rv) | `src/market_data/vrp_loaders.py` |
| VRP Core features (VIX, VX1/2/3, ES returns from market) | `src/agents/feature_vrp_core.py` |
| VRP Convergence features (VIX, VX1/2) | `src/agents/feature_vrp_convergence.py` |
| VRP Alt features (VIX, VX1, ES returns from market) | `src/agents/feature_vrp_alt.py` |
| VRP stress proxy (VVIX, gamma_stress, vx_backwardation) | `src/agents/feature_vrp_stress.py` |
| VX backwardation (VX1 vs VX2) | `src/agents/feature_vx_backwardation.py` |
| Policy feature builder (VIX, VIX3M, VVIX, VX curve, vx_backwardation, vrp_stress) | `src/agents/policy_feature_builder.py` |
| MarketData table discovery & universe | `src/agents/utils_db.py` (`find_ohlcv_table`), `src/agents/data_broker.py` |
| Verify_setup (table name printed) | `scripts/verify_setup.py` |
| Audit (g_continuous_bar_daily, market_data, f_fred) | `scripts/diagnostics/audit_carry_inputs_coverage.py` |
| VRP Phase-1 diagnostics (VX1 returns from market_data) | `src/diagnostics/vrp_core_phase1.py`, `src/diagnostics/vrp_convergence_phase1.py` |
| VX curve carry / sanity | `src/strategies/carry/vx_calendar_carry.py`, `src/diagnostics/vx_carry_sanity.py` |
| Config (DB path, universe) | `configs/data.yaml` |
| SOT: VRP data sources | `docs/SOTs/STRATEGY.md` (VRP Data Requirements) |

---

## 5. Checklist: Minimal ingestion for VRP signals

- [ ] **f_fred_observations:** `VIXCLS` daily; optional `VVIXCLS`.
- [ ] **market_data_cboe:** `VIX3M`, `VVIX` daily settle.
- [ ] **market_data:** `@VX=101XN`, `@VX=201XN`, `@VX=301XN` daily close.
- [ ] **OHLCV table (e.g. g_continuous_bar_daily):** `ES_FRONT_CALENDAR_2D` present so MarketData provides ES returns for RV21 (Core) and RV5 (Alt).
- [ ] All above in the **same** DuckDB pointed to by `configs/data.yaml` → `db.path` (or strategy `db_path`).

This is the minimal set so that VRP Core, Convergence, and Alt produce signals on every rebalance date, and policy gating (vx_backwardation, vrp_stress_proxy) works.
