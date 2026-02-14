# Post–Risk-Targeting Normalization Audit

**Scope:** Single rebalance cycle in ExecSim: from `RiskTargetingLayer.scale_weights()` to daily P&L.

**Conclusion:** **PASS** — RT-scaled weights are used linearly through to P&L. No renormalization, re-budgeting, or exposure rescaling occurs after RT.

---

## 1. Where RT scaling is called

| Item | Detail |
|------|--------|
| **File** | `src/agents/exec_sim.py` |
| **Line** | 987 |
| **Call** | `weights_raw = risk_targeting.scale_weights(weights=weights_pre_rt, returns=returns_simple, date=date)` |
| **Variable receiving scaled weights** | `weights_raw` |
| **Stored or transformed** | Stored; then either copied or multiplied by a scalar (allocator). No normalization. |

---

## 2. Call chain: RT output → P&L

```
RiskTargetingLayer.scale_weights(...)
    → returns: scaled weights (unit_weights * leverage, with optional safety renormalization inside RT)
    ↓
weights_raw = <return value>                    [exec_sim.py:987]
    ↓
[Optional] weights = weights_raw * risk_scalar_applied   [exec_sim.py:1198]  (allocator Layer 6; 0 < scalar ≤ 1)
    else       weights = weights_raw.copy()               [exec_sim.py:1201]
    ↓
weights_history.append(weights)                [exec_sim.py:1205]
    ↓
weights_panel = pd.DataFrame(weights_history, index=rebalance_dates[:len(weights_history)])  [exec_sim.py:1346]
    ↓
weights_daily = weights_panel.reindex(returns_df.index).ffill().fillna(0.0)   [exec_sim.py:1367]
    ↓
weights_aligned = weights_daily[common_symbols]   [exec_sim.py:1372]  (column subset only)
    ↓
portfolio_returns_log = (weights_aligned * returns_aligned).sum(axis=1)   [exec_sim.py:1376]
portfolio_returns_daily = np.exp(portfolio_returns_log) - 1.0              [exec_sim.py:1377]
```

**P&L computation (exact location):**  
- **Function:** `ExecSim.run()` (in-place); same logic appears in `_save_run_artifacts()` for artifact path.  
- **Lines:** 1376–1377 (run path), 1716–1717 (artifact path).  
- **Formula:** `portfolio_returns_log = (weights_aligned * returns_aligned).sum(axis=1)` then convert log → simple.  
- **Weight variable used:** `weights_aligned`, which is a column subset of `weights_daily`, which is forward-filled `weights_panel`, which is built from `weights` (RT output, optionally × allocator scalar).

---

## 3. Variable names at each step

| Step | Variable | Type | Notes |
|------|----------|------|--------|
| 1 | `weights_pre_rt` | pd.Series | Input to RT (allocator output). |
| 2 | `weights_raw` | pd.Series | Output of `scale_weights()`; **RT-scaled weights**. |
| 3 | `weights` | pd.Series | Either `weights_raw` or `weights_raw * risk_scalar_applied`. |
| 4 | `weights_history` | list of Series | One Series per rebalance date. |
| 5 | `weights_panel` | pd.DataFrame | Index = rebalance dates; columns = symbols. |
| 6 | `weights_daily` | pd.DataFrame | `weights_panel` reindexed to daily, ffilled, fillna(0). |
| 7 | `weights_aligned` | pd.DataFrame | `weights_daily[common_symbols]` (alignment only). |
| 8 | P&L | `portfolio_returns_log` / `portfolio_returns_daily` | `(weights_aligned * returns_aligned).sum(axis=1)` then exp−1. |

---

## 4. Search for post-RT normalization / rescaling

Searched the codebase for:

- `weights = weights / ...`, `weights /= ...`, `weights / gross_exposure`, `weights / weights.sum()`
- Implicit rebudgeting across sleeves
- Re-scaling to target 1.0
- Clipping that could materially reduce exposure
- Instrument-level caps, per-asset max weight
- Portfolio construction overwriting scaled weights

**Findings:**

- **Inside RT only:** In `src/layers/risk_targeting.py` (lines 423–440), `unit_weights = weights / gross_exposure` and optional `scale = leverage_cap / gross_after` are applied **inside** `scale_weights()` before returning. The **return value** of `scale_weights()` is already the final scaled weights (no further normalization in ExecSim).
- **ExecSim after RT:**  
  - **Allocator (Layer 6):** `weights = weights_raw * risk_scalar_applied` (line 1198). This is a single scalar multiplication (0 < scalar ≤ 1), **not** a normalization; gross exposure is scaled down proportionally.  
  - **Panel construction:** `weights_panel = pd.DataFrame(weights_history, ...)` — no division.  
  - **Daily expansion:** `weights_daily = weights_panel.reindex(...).ffill().fillna(0.0)` — index alignment and forward fill only; `fillna(0.0)` only for dates before first rebalance. **No division by sum or gross.**  
  - **Column alignment:** `weights_aligned = weights_daily[common_symbols]` — column subset only.  
- **P&L:** `(weights_aligned * returns_aligned).sum(axis=1)` — linear in weights; `.sum(axis=1)` sums over assets (columns), it does not normalize weights.

No other code path in ExecSim overwrites or renormalizes the RT (or post-allocator) weights before P&L.

---

## 5. Explicit answers

| Question | Answer |
|----------|--------|
| **Are RT-scaled weights the exact weights used in P&L?** | Yes, up to the optional allocator scalar. The weights used in P&L are either `weights_raw` (RT output) or `weights_raw * risk_scalar_applied`. Both are stored in `weights_history` → `weights_panel` → `weights_daily` → `weights_aligned` and used in the product with returns. |
| **Is there ANY renormalization step after RT?** | No. No division by sum, gross, or any normalizer after `scale_weights()` returns. |
| **Is there any transformation that could neutralize leverage scaling?** | No. The only post-RT transformation is optional multiplication by `risk_scalar_applied` (≤ 1), which scales exposure down, and index/column alignment (reindex, ffill, subset). None of these renormalize or undo RT leverage. |
| **Full linearity from RT output to P&L?** | Yes. From RT output (`weights_raw`) to P&L: only scalar multiplication (allocator), stacking into a panel, reindex/ffill, column subset, and then linear formula `(w * r).sum(axis=1)`. Scaling is preserved. |

---

## 6. Conclusion

**PASS.** No post-RT normalization, re-budgeting, or exposure rescaling was found. The weight lifecycle from `scale_weights()` to daily P&L is linear and preserves RT (and allocator) scaling.
