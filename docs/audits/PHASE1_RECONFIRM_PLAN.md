# Phase-1 Reconfirmation Sweep — Plan

**Purpose:** Reconfirm each sleeve under the institutional Math Contract (simple returns, lag-1 weights, 252 trading-day annualization) without changing strategy economics or sleeve implementations.

**Canonical window:** 2020-01-06 → 2025-10-31  
**Universe source:** Canonical universe from `configs/data.yaml` (13 continuous + VX from vx_universe = 16 total). Logged per run in meta.json.  
**Run ID format:** `phase1_reconfirm_<sleeve_name>_simplepnl_20200106_20251031_<YYYYMMDD_HHMMSS>`

---

## Source of truth: sleeve list

Sleeves are taken from **configs/pinned_runs.yaml** (Core V9 metasleeves) and **run_strategy.py** strategy keys (internal names). Each reconfirmation runs a **single sleeve at weight 1.0** with all others explicitly disabled.

| # | Sleeve (internal name)     | Type     | Metasleeve (display) | Min symbols (strict_universe) | Harness |
|---|----------------------------|----------|----------------------|-------------------------------|---------|
| 1 | tsmom_multihorizon         | metasleeve | trend              | 16 (13 non-VX + 3 VX for policy/vol) | Single-sleeve config, RT on, allocator off |
| 2 | csmom_meta                 | metasleeve | csmom              | 16                             | Same |
| 3 | vrp_core_meta              | atomic     | vrp                | 16 (VRP trades VX1–VX3 only)   | Same |
| 4 | vrp_convergence_meta       | atomic     | vrp                | 16                             | Same |
| 5 | vrp_alt_meta               | atomic     | vrp                | 16                             | Same |
| 6 | sr3_curve_rv_meta          | metasleeve | curve_rv           | 16 (SR3 + rates)                | Same |
| 7 | vx_calendar_carry          | metasleeve | vx_carry           | 16 (VX legs)                    | Same |

**Total: 7 sleeves.**

---

## Execution

- **Base config:** `configs/strategies.yaml` (full strategies + engine_policy, risk_targeting, allocator_v1, exec, risk_vol).
- **Override per sleeve:** In-memory (or temp file) merge:
  - `strategies`: only `<sleeve>` enabled with `weight: 1.0` and same `params` as in base; all other sleeves `enabled: false`, `weight: 0`.
  - `engine_policy_v1.enabled`: false (no policy gating for Phase-1 isolation).
  - `macro_regime.enabled`: false.
  - `risk_targeting`: enabled, target_vol 0.20, vol_lookback 63, leverage_cap 7.0, leverage_floor 1.0, vol_floor 0.05 (Phase-1 default).
  - `allocator_v1`: enabled, mode `"off"` (no precomputed scaling).
- **Command:**  
  `python3 run_strategy.py --config_path <merged_config_path> --run_id <run_id> --start 2020-01-06 --end 2025-10-31 --strict_universe`  
  (No `--strategy_profile`; strategies come from merged config.)
- **Post-run:** Ensure attribution (regenerate if needed), ensure `leverage_summary.json` or risk scalars exist, record metrics from `meta.json` into sweep results. Run `extract_metrics.py --run_id` only for runs that are later pinned (see Step E).

---

## Outputs

- **Sweep results:** `docs/audits/PHASE1_RECONFIRM_SWEEP_RESULTS.csv`, `docs/audits/PHASE1_RECONFIRM_SWEEP_RESULTS.md`
- **Per-sleeve summaries:** `docs/audits/phase1_reconfirm/<sleeve>.md`
- **Historical map:** `docs/audits/PHASE1_RECONFIRM_HISTORICAL_MAP.md` (Step C)

---

## Notes

- If a sleeve fails (missing data, strict_universe failure), record as **SKIPPED** with reason; do not change sleeve code.
- Confirm each run’s `meta.json` includes `pnl_contract` (simple, lag1) after run.
- Universe used is logged in meta.json (`universe` list length = 16 for full canonical).
