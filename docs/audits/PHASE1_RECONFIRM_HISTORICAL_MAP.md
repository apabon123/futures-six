# Phase-1 Reconfirmation — Historical Phase-1 Artifact Map

Best-effort mapping from **reconfirm sleeve** (internal name) to historical Phase-1 artifacts.  
Used to compare current reconfirm metrics vs historical Phase-1 (window/contract may differ).

**Contract note:** Historical Phase-1 runs may use legacy PnL or unknown contract unless marked otherwise.

---

## Sleeve → Historical artifact

| Sleeve (reconfirm)     | Historical Phase-1 artifact | run_id / path | Window | Metrics in file | Note |
|-----------------------|-----------------------------|---------------|--------|-----------------|------|
| tsmom_multihorizon    | trend/medium_canonical/phase1.txt | med_canonical_phase1_standalone_20251119_115012 | 2018-01-01 to 2025-10-31 | No | Proxy: medium-term canonical (part of multihorizon). Legacy run_id in phase1.txt. |
| tsmom_multihorizon    | trend/short_canonical/phase1.txt  | short_canonical_phase1_standalone_20251119_160802 | 2018-01-01 to 2025-10-31 | No | Proxy: short-term canonical. |
| csmom_meta            | **NOT FOUND**               | — | — | — | csmom has Phase-2 only (phase2.txt); no Phase-1 index. |
| vrp_core_meta         | vrp/vrp_core_phase1.txt     | path: data\\diagnostics\\vrp_core_phase1\\20251209_214729 | 2020-01-01 to 2025-10-31 | sharpe 0.4045, cagr 0.0827, max_dd -0.3331 | Legacy path format. |
| vrp_convergence_meta  | vrp/vrp_convergence_phase1.txt | path: data\\diagnostics\\vrp_convergence_phase1\\20251211_093401 | 2020-01-01 to 2025-10-31 | sharpe 0.2671, cagr 0.0033, max_dd -0.0208 | Legacy path format. |
| vrp_alt_meta          | vrp/vrp_alt/phase1.txt      | path: data\\diagnostics\\vrp_alt_phase1\\20251213_123417 | 2020-01-01 to 2025-10-31 | sharpe 0.9142, cagr 0.0183, max_dd -0.0201 | Legacy path format. |
| sr3_curve_rv_meta     | rates_curve_rv/sr3_curve_rv_pack_slope_momentum/phase1.txt | 20251217_134842 | — | No | Proxy: pack_slope atomic. run_id in file. |
| sr3_curve_rv_meta     | rates_curve_rv/sr3_curve_rv_rank_fly_2_6_10_momentum/phase1.txt | 20251217_134842 | — | No | Proxy: rank_fly atomic. |
| vx_calendar_carry     | carry/vx_calendar_carry/vx2_vx1_short/phase1.txt | vxcarry_phase1_vx2_vx1_short_20251217_125402 | — | No | VX2-VX1 short variant. |

---

## Summary

- **Sleeves with at least one historical Phase-1 reference:** 6 (trend proxy, vrp_core, vrp_convergence, vrp_alt, sr3_curve_rv proxy, vx_calendar_carry).
- **Sleeve with no Phase-1 index:** csmom_meta (Phase-2 only).
- **Window alignment:** Reconfirm window is 2020-01-06 → 2025-10-31; some historical use 2020-01-01 → 2025-10-31 or 2018 start; compare with caution.
- **Contract:** Historical artifacts are legacy or unknown; reconfirm runs use explicit simple/lag1 contract.
