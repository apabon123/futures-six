# Allocator v1: Quick Start Guide

**Version:** v1.0  
**Status:** ✅ Production Ready (Stages 4A-4D Complete)

---

## What is Allocator v1?

Allocator v1 is a **state-regime-risk system** for portfolio risk management. It:

1. **Computes 10 state features** from portfolio/asset data (volatility, drawdown, correlation, etc.)
2. **Classifies market regimes** (NORMAL, ELEVATED, STRESS, CRISIS) using rule-based logic
3. **Generates risk scalars** (0.25-1.0) for portfolio exposure adjustment
4. **Saves all artifacts** automatically for analysis and tuning

**Key Benefits:**
- ✅ Deterministic and auditable (no black-box ML)
- ✅ Composable (state → regime → risk layers are independent)
- ✅ Sticky (hysteresis prevents regime thrashing)
- ✅ Parameter-light (few knobs, sensible defaults)
- ✅ No rewiring (enable/disable with one config flag)

---

## Quick Start: Run a Backtest

### 1. Run Strategy with Allocator v1 Artifacts

```bash
python run_strategy.py \
  --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \
  --start 2024-01-01 \
  --end 2024-12-15 \
  --run_id my_test_run
```

**What Happens:**
- Backtest runs normally (weights not modified yet)
- All allocator v1 artifacts are computed and saved automatically
- Check `reports/runs/my_test_run/` for artifacts

### 2. Verify Artifacts Were Created

```bash
ls reports/runs/my_test_run/allocator_*
```

**Expected Files:**
```
allocator_state_v1.csv           # 10 state features
allocator_state_v1_meta.json     # State metadata
allocator_regime_v1.csv          # Daily regime series
allocator_regime_v1_meta.json    # Regime metadata
allocator_risk_v1.csv            # Daily risk scalar
allocator_risk_v1_meta.json      # Risk metadata
```

### 3. Inspect Regime Distribution

```bash
python scripts/diagnostics/run_allocator_regime_v1.py --run_id my_test_run
```

**Example Output:**
```
Regime Distribution:
  NORMAL    : 144 days (60.8%), max consecutive: 59 days
  ELEVATED  :  88 days (37.1%), max consecutive: 33 days
  STRESS    :   5 days ( 2.1%), max consecutive:  5 days
  CRISIS    :   0 days ( 0.0%)
```

---

## Understanding the Artifacts

### State Features (`allocator_state_v1.csv`)

10 features computed from portfolio data:

**Volatility / Acceleration:**
- `port_rvol_20d`: 20-day realized volatility
- `port_rvol_60d`: 60-day realized volatility
- `vol_accel`: Ratio of short-term to long-term volatility

**Drawdown / Path:**
- `dd_level`: Current drawdown from peak
- `dd_slope_10d`: 10-day change in drawdown

**Cross-Asset Correlation:**
- `corr_20d`: 20-day average pairwise correlation
- `corr_60d`: 60-day average pairwise correlation
- `corr_shock`: Difference between short-term and long-term correlation

**Engine Health:**
- `trend_breadth_20d`: Fraction of trend signals positive over 20 days
- `sleeve_concentration_60d`: Herfindahl index of sleeve PnL concentration

### Regimes (`allocator_regime_v1.csv`)

4 regime classifications:

| Regime | Description | Typical Risk Scalar |
|--------|-------------|---------------------|
| **NORMAL** | Typical market conditions | 1.00 (no adjustment) |
| **ELEVATED** | Increased volatility or correlation | 0.85 (moderate reduction) |
| **STRESS** | Significant drawdown or volatility spike | 0.55 (significant reduction) |
| **CRISIS** | Extreme conditions | 0.30 (defensive positioning) |

**Regime Logic:**
- Rule-based classification using state features
- Hysteresis: separate enter/exit thresholds to prevent thrashing
- Anti-thrash: minimum 5 days in regime before downgrade

### Risk Scalars (`allocator_risk_v1.csv`)

Portfolio-level exposure scalar:
- **Range:** [0.25, 1.0]
- **Smoothing:** EWMA with alpha=0.25 (5-day half-life)
- **Mapping:** Regime → target scalar → smoothed output

**Example:**
```
Date        risk_scalar
2024-01-02  1.000      # NORMAL
2024-02-15  0.850      # ELEVATED
2024-03-10  0.550      # STRESS
2024-04-01  0.750      # Recovering (smoothed)
```

---

## Configuration

### Enable/Disable Allocator v1

Edit `configs/strategies.yaml`:

```yaml
allocator_v1:
  enabled: false  # Set to true to apply risk scalars (not yet implemented)
  state_version: "v1.0"
  regime_version: "v1.0"
  risk_version: "v1.0"
```

**Important:**
- Artifacts are **always computed and saved** (regardless of `enabled` flag)
- The `enabled` flag will control risk scalar application (future feature)
- Current implementation: artifacts saved for analysis, weights not modified

---

## Analysis Workflow

### Load Artifacts in Python

```python
import pandas as pd
import json

run_id = "my_test_run"
run_dir = f"reports/runs/{run_id}"

# Load state
state_df = pd.read_csv(f"{run_dir}/allocator_state_v1.csv", index_col=0, parse_dates=True)

# Load regime
regime_df = pd.read_csv(f"{run_dir}/allocator_regime_v1.csv", index_col=0, parse_dates=True)
regime = regime_df['regime']

# Load risk
risk_df = pd.read_csv(f"{run_dir}/allocator_risk_v1.csv", index_col=0, parse_dates=True)
risk_scalar = risk_df['risk_scalar']

# Load metadata
with open(f"{run_dir}/allocator_state_v1_meta.json") as f:
    state_meta = json.load(f)
with open(f"{run_dir}/allocator_regime_v1_meta.json") as f:
    regime_meta = json.load(f)
with open(f"{run_dir}/allocator_risk_v1_meta.json") as f:
    risk_meta = json.load(f)
```

### Analyze Regime Transitions

```python
# Regime distribution
print(regime.value_counts())

# Transition matrix
transitions = {}
for i in range(len(regime) - 1):
    from_regime = regime.iloc[i]
    to_regime = regime.iloc[i + 1]
    key = f"{from_regime}->{to_regime}"
    transitions[key] = transitions.get(key, 0) + 1

print("Top Transitions:")
for trans, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {trans:25s}: {count:3d}")
```

### Plot Risk Scalar Over Time

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(risk_scalar.index, risk_scalar.values, label='Risk Scalar', linewidth=2)
ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Full Risk')
ax.axhline(0.85, color='yellow', linestyle='--', alpha=0.5, label='ELEVATED')
ax.axhline(0.55, color='orange', linestyle='--', alpha=0.5, label='STRESS')
ax.axhline(0.30, color='red', linestyle='--', alpha=0.5, label='CRISIS')
ax.set_xlabel('Date')
ax.set_ylabel('Risk Scalar')
ax.set_title('Allocator v1: Risk Scalar Over Time')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{run_dir}/risk_scalar_plot.png", dpi=150)
plt.show()
```

### Compare Regime vs. Drawdown

```python
# Load portfolio returns
portfolio_returns = pd.read_csv(f"{run_dir}/portfolio_returns.csv", index_col=0, parse_dates=True)['ret']

# Compute equity curve
equity = (1 + portfolio_returns).cumprod()

# Compute drawdown
running_max = equity.expanding().max()
drawdown = (equity / running_max) - 1

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Equity and drawdown
ax1.plot(equity.index, equity.values, label='Equity Curve', linewidth=2)
ax1.set_ylabel('Equity')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.fill_between(drawdown.index, 0, drawdown.values, alpha=0.3, color='red', label='Drawdown')
ax2.set_ylabel('Drawdown')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(alpha=0.3)

# Overlay regime colors
regime_colors = {'NORMAL': 'green', 'ELEVATED': 'yellow', 'STRESS': 'orange', 'CRISIS': 'red'}
for i in range(len(regime) - 1):
    date_start = regime.index[i]
    date_end = regime.index[i + 1]
    regime_name = regime.iloc[i]
    color = regime_colors.get(regime_name, 'gray')
    ax2.axvspan(date_start, date_end, alpha=0.2, color=color)

plt.tight_layout()
plt.savefig(f"{run_dir}/regime_vs_drawdown.png", dpi=150)
plt.show()
```

---

## Troubleshooting

### Issue: Empty State DataFrame

**Symptom:**
```
[ExecSim] ⚠️  Allocator state computation returned empty DataFrame
```

**Cause:** Not enough data for rolling window calculations (need at least 60 days)

**Solution:** Use a longer date range (e.g., start from 2024-01-01 instead of 2024-11-01)

### Issue: Missing Optional Features

**Symptom:**
```
features_missing: ["trend_breadth_20d", "sleeve_concentration_60d"]
```

**Cause:** Trend unit returns or sleeve returns not available

**Solution:** This is expected if:
- Strategy has no Trend sleeve (trend_breadth_20d will be missing)
- Strategy has only one sleeve (sleeve_concentration_60d will be missing)

**Note:** The system will still work with only the 8 required features.

### Issue: All Regimes are NORMAL

**Symptom:**
```
Regime Distribution:
  NORMAL: 100%
```

**Cause:** Market conditions were benign during backtest period

**Solution:** This is expected behavior. Try a longer date range or a period with known volatility (e.g., 2020, 2022).

---

## Next Steps

1. **Run on Full Historical Data:**
   ```bash
   python run_strategy.py --strategy_profile <profile> --start 2015-01-01 --end 2024-12-15 --run_id full_history
   ```

2. **Analyze Regime Behavior During Known Events:**
   - COVID crash (March 2020)
   - 2022 selloff (Feb-Oct 2022)
   - 2023 banking crisis (March 2023)

3. **Tune Thresholds (if needed):**
   - Edit `src/allocator/regime_rules_v1.py`
   - Adjust enter/exit thresholds based on historical regime transitions
   - Re-run diagnostic scripts to regenerate artifacts

4. **Implement Risk Scalar Application (Stage 5):**
   - Requires lagged or rolling window approach to avoid circular dependency
   - See `docs/ALLOCATOR_V1_STAGE_4_COMPLETE.md` for details

---

## Reference

- **Full Documentation:** `docs/ALLOCATOR_V1_STAGE_4_COMPLETE.md`
- **Implementation Details:** `docs/STAGE_4_IMPLEMENTATION.md`
- **Original Build Plan:** `docs/ALLOCATOR_STATE_V1_FINALIZED.md`

---

## Support

For questions or issues, refer to:
1. Comprehensive documentation in `docs/`
2. Code comments in `src/allocator/`
3. Diagnostic scripts in `scripts/diagnostics/`

**Key Files:**
- `src/allocator/state_v1.py` - State feature computation
- `src/allocator/regime_v1.py` - Regime classification
- `src/allocator/risk_v1.py` - Risk scalar transformation
- `src/agents/exec_sim.py` - Integration point (search for "Stage 4")

