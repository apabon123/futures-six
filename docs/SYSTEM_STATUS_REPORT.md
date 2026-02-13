# System Status Report

**Repo:** futures-six  
**Branch:** main (unchanged)  
**Date:** 2026-02-05  

---

## 1. Canonical Evaluation Window

**Answer:** **2020-01-06** to **2025-10-31**

| Source | Location |
|--------|----------|
| **Config (authoritative)** | `configs/canonical_window.yaml` lines 4–6: `canonical_window.start_date`, `canonical_window.end_date` |
| **Code loader** | `src/utils/canonical_window.py`: `load_canonical_window()` reads that YAML and returns `(start_date, end_date)` |
| **Docs** | `docs/SOTs/STRATEGY.md` § "Canonical Evaluation Window" (lines 55–68): defines window and states all canonical metrics must use it |

All canonical performance metrics are defined over this window. Use `src.utils.canonical_window.load_canonical_window()` in code; see also `src/config/backtest_window.py` (lines 4–9) for the legacy note pointing to this config.

---

## 2. Active Sleeves in Strategy Profile: `core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro`

**Profile:** Core v9 (production baseline).  
**Definition:** `configs/strategies.yaml` lines 1376–1485.

### Sleeves (all enabled)

| Sleeve | Weight | Config key | Notes |
|--------|--------|------------|--------|
| Trend (TSMOM multihorizon) | 52.44% | `tsmom_multihorizon` | Long 252 / med 84 / short 21; vol normalization, rebalance W-FRI |
| CSMOM | 21.85% | `csmom_meta` | Lookbacks 63/126/252, cross-section neutralized |
| VRP-Core | 6.555% | `vrp_core_meta` | RV21, zscore, signal_mode zscore |
| VRP-Convergence | 2.185% | `vrp_convergence_meta` | zscore, target_vol 0.10 |
| VRP-Alt | 13.11% | `vrp_alt_meta` | zscore, target_vol 0.10 |
| VX Calendar Carry | 4.6% | `vx_calendar_carry` | variant `vx2_vx1_short` |
| Curve RV (SR3) | 8% | `sr3_curve_rv_meta` | Rank Fly 5% + Pack Slope 3% (`rank_fly`, `pack_slope`) |

**Components (meta-level):** `trend_meta`, `csmom_meta`, `vrp_core`, `vrp_convergence`, `vrp_alt`, `vx_carry`, `curve_rv` all `true`.  
**Overlays:** `macro_regime: false` (no macro overlay).

### Key switches (from config and run meta)

- **Allocator v1** (`configs/strategies.yaml` § allocator_v1, lines 96–125): Default `mode: "off"` (artifacts only). For production, use `mode: "precomputed"` with `precomputed_run_id` set. **Profile H** is the recommended production profile.
- **Risk targeting** (`configs/strategies.yaml` lines 80–88): **Enabled**; `target_vol: 0.20` (baseline); vol lookback 63, leverage cap 7.0, floor 1.0. Frozen Phase 3B runs use `target_vol: 0.42` (see `reports/_PINNED/README.md`).
- **Engine policy v1** (lines 38–73): Default `enabled: false`, `mode: "off"`. Phase 3A+ baselines use policy (e.g. trend/VRP gating) from a compute baseline.

Run-specific behavior is in each run’s `meta.json`: `allocator_v1.mode` (`off` | `precomputed`), `allocator_v1.profile`, `risk_targeting.effective`, `risk_targeting.target_vol` (when set).

---

## 3. Latest Baseline run_id (Current Reference Baseline)

**From `reports/_PINNED/README.md` and `reports/runs/`:**

- **Phase 3B (current golden proof):** The documented reference baseline pair is the **Phase 3B Integrity Baseline v1 (2026-01-20)**:
  - **Traded (post-allocator) baseline:** `phase3b_baseline_traded_20260120_093953`
  - **Artifacts-only baseline:** `phase3b_baseline_artifacts_only_20260120_093953`

- **Phase 3A statistical baseline:** `phase3a_statistical_baseline_governed_20251031` (evaluation 2020-03-20 to 2025-10-31).

- **Phase 3B post-allocator canonical:** `phase3b_post_allocator_canonical_baseline_20260116` (also documented in _PINNED).

**Recommendation:** Treat **`phase3b_baseline_traded_20260120_093953`** as the **current reference baseline** for:
- Attribution and regression checks  
- Engine-quality and calibration work (per Phase 3B close-out and _PINNED README)

It has canonical window dates, all 7 Phase 3B checkpoints passed, and represents the traded (post-allocator) curve. If you need a pre-allocator reference, use **`phase3b_baseline_artifacts_only_20260120_093953`**.

There is no single “official baseline run_id” field in the SOTs; the above is the recommended pin from `reports/_PINNED/README.md`.

---

## 4. Performance Summary

### 4.1 y2023_2024_sanity_v1 (already run)

**Source:** `reports/runs/y2023_2024_sanity_v1/meta.json`

| Field | Value |
|-------|--------|
| **run_id** | y2023_2024_sanity_v1 |
| **Window** | 2023-01-01 to 2024-12-31 (effective_start 2023-01-06) |
| **canonical_window** | **false** (not the canonical 2020-01-06 → 2025-10-31 window) |
| **Strategy profile** | core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro |

**Metrics (eval window):**

| Metric | Value |
|--------|--------|
| CAGR | ~0.42% (metrics_eval.cagr ≈ 0.00418) |
| Vol | ~8.13% |
| Sharpe | ~0.082 |
| Max drawdown | ~-9.72% |
| Hit rate | ~52.5% |
| Avg turnover | ~0.59 |
| Avg gross | ~2.58× |
| Avg net | ~0.46× |
| n_periods | 104 |

**Governance:** Risk targeting effective; allocator_v1 enabled but **mode "off"** (effective false, inputs_missing true). So this run is **RT-only, no allocator scaling**.

---

### 4.2 Canonical baseline run_id referenced in docs/SOTs

Docs and _PINNED refer to several baselines; the one with full canonical window and governed metrics is the **Phase 3A statistical baseline** and the **Phase 3B traded** run. Summaries from `meta.json`:

**Phase 3A statistical baseline:** `phase3a_statistical_baseline_governed_20251031`

- **Window:** 2020-01-06 to 2025-10-31 (data); **evaluation:** 2020-03-20 to 2025-10-31 (effective_start).
- **canonical_window:** true  
- **Metrics (eval):** CAGR ~7.18%, Vol ~8.86%, **Sharpe ~0.676**, MaxDD ~-12.0%, Hit rate ~52.9%.
- **Allocator:** mode precomputed, effective true, profile H. RT effective.

**Phase 3B traded baseline:** `phase3b_baseline_traded_20260120_093953` (recommended reference)

- **Window:** 2020-01-06 to 2025-10-31; **evaluation:** 2020-03-20 to 2025-10-31.
- **canonical_window:** true  
- **Metrics (eval):** CAGR ~16.34%, Vol ~16.68%, **Sharpe ~0.815**, MaxDD ~-17.84%, Hit rate ~52.4%.
- **Governance:** RT target_vol 0.42; allocator v1 precomputed, applied, canonical_mode "applied".

*(STRATEGY.md lines 352, 361 cite Core v8/v9 on canonical window: Core v8 Sharpe 0.5820, Core v9 Sharpe 0.6605 — those are pre-allocator/earlier stack; the Phase 3B traded run is the current post-allocator reference.)*

---

## 5. ROADMAP.md: Phase, Acceptance Criteria, Next Steps

**Source:** `docs/SOTs/ROADMAP.md`

### Current phase

- **Phase 3A:** COMPLETE (Jan 2026): policy features, governed re-freeze, RT & allocator governance, pinned baseline `phase3a_statistical_baseline_governed_20251031`.
- **Phase 3B:** COMPLETE (Close-Out 2026-01-20): integrity baseline pair frozen, 7 checkpoints passed, construction/allocator decoupling verified. Ready for Phase 4.

So the project is **between Phase 3B and Phase 4**: Phase 3B is done; **Phase 4 (Engine Quality)** is the stated next focus (see _PINNED README “Next Phase (Phase 4 — Engine Quality)” and ROADMAP §3 “Next Steps”).

### Acceptance criteria remaining

- **Phase 3A attribution ablations:** Not yet done; deferred until RT/Allocator behavior is frozen (ROADMAP lines 114–115).
- **Phase 4 engine quality (from _PINNED and ROADMAP):** Post-construction Sharpe > 0.5 with stable regime behavior; use attribution to prioritize engine work; improve individual sleeve Sharpes at post-construction.
- **Calibration sprint (ROADMAP lines 107–111):** RT + Allocator calibration (distribution targets: median gross 4.0–4.5×, P90 5.5–6.5×, cap rarely binds).

### Next 3 concrete steps (from ROADMAP and _PINNED)

1. **RT + Allocator calibration** — Run calibration sprint: hit distribution targets (median gross 4.0–4.5×, P90 gross 5.5–6.5×, cap rarely binds); freeze RT/allocator behavior so ablations are valid (ROADMAP §3 “Next Steps (Calibration Sprint)”).

2. **Phase 3A attribution ablations** — Once RT/allocator is frozen: run Policy / RT / Allocator / Sleeves ablations per committee-pack workflow (ROADMAP line 115; _PINNED “Next: Phase 3A Attribution Ablations”).

3. **Phase 4 engine quality** — Use Phase 3B post-construction attribution to prioritize sleeves; improve post-construction sleeve Sharpes; target post-construction Sharpe > 0.5 and stable regime behavior (reports/_PINNED/README.md “Next Phase (Phase 4 — Engine Quality)”).

---

*Report produced from code, configs, docs, and `reports/runs/` meta.json only. No backtests were run.*
