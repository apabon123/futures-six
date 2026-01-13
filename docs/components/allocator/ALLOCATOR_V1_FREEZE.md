# Allocator v1 — Production Freeze

**Version:** v1.0  
**Freeze Date:** December 18, 2024  
**Status:** FROZEN (Production-Ready)

---

## What Allocator v1 Is

Allocator v1 is a **rule-based, deterministic risk control system** that scales portfolio exposure based on market conditions.

**Core Function:**
- Observes portfolio state (volatility, drawdown, correlation)
- Classifies risk regimes (NORMAL, ELEVATED, STRESS, CRISIS)
- Computes portfolio-level risk scalars (0.25–1.0)
- Scales weights with 1-rebalance lag

**Architecture (Frozen):**
1. **State Layer**: 10 features (volatility, drawdown, correlation, engine health)
2. **Regime Layer**: 4 regimes (rule-based, hysteresis, anti-thrash)
3. **Risk Layer**: Risk scalars (monotonic regime mapping, EWMA smoothed)
4. **Exposure Layer**: Weight scaling (1-rebalance lag, no circularity)

**Production Mode (Frozen):**
```yaml
allocator_v1:
  enabled: true
  mode: "precomputed"  # PRODUCTION DEFAULT
  precomputed_run_id: "<validated_baseline_run_id>"
```

**Key Properties:**
- ✅ Deterministic (same inputs → same outputs)
- ✅ Auditable (complete artifact trail)
- ✅ Portfolio-level only (no sleeve-specific scaling)
- ✅ Risk governor (reduces tail risk, preserves NORMAL returns)
- ✅ No machine learning (rule-based only)
- ✅ No optimization (fixed thresholds)

---

## What Allocator v1 Is NOT

**v1 explicitly does NOT:**
- ❌ **Optimize for Sharpe** (it optimizes for survivability)
- ❌ **Predict markets** (it reacts to realized conditions)
- ❌ **Generate returns** (it controls risk)
- ❌ **Use machine learning** (rule-based only)
- ❌ **Scale sleeves individually** (portfolio-level only)
- ❌ **Activate convexity overlays** (always-on passive scaling only)
- ❌ **Tune thresholds dynamically** (fixed regime thresholds)
- ❌ **Learn from P&L** (no feedback loops)

**Design Philosophy:**

Allocator v1 is intentionally **simple, transparent, and conservative**. It is a risk governor, not an alpha engine.

---

## Production Mode: `precomputed` (Locked)

**Production deployments MUST use `mode="precomputed"`.**

**Why Precomputed:**
- No warmup period issues
- No circular dependencies
- Fully deterministic
- Complete audit trail
- Institutional standard

**Other Modes:**
- **`mode="off"`**: Baseline generation only (allocator computes artifacts but doesn't scale)
- **`mode="compute"`**: Research-only (has warmup issues, not production-safe until Stage 9)

**Deployment Workflow:**
1. Generate baseline run with `enabled=false` (Pass 1)
2. Validate with Stage 6.5 checklist
3. Set `precomputed_run_id` to validated baseline
4. Deploy with `enabled=true, mode="precomputed"` (Pass 2)

---

## Frozen Components (v1.0)

**The following are FROZEN and require v2 for changes:**

### 1. State Features (10 features)
- `port_rvol_20d`, `port_rvol_60d`, `vol_accel`
- `dd_level`, `dd_slope_10d`
- `corr_20d`, `corr_60d`, `corr_shock`
- `trend_breadth_20d`, `sleeve_concentration_60d`

**Frozen:** Feature definitions, lookback windows, calculations

### 2. Regime Logic (4 regimes)
- NORMAL, ELEVATED, STRESS, CRISIS
- Rule-based classification (stress conditions + risk score)
- Hysteresis (separate enter/exit thresholds)
- Anti-thrash (5-day minimum regime duration)

**Frozen:** Regime definitions, classification logic, hysteresis mechanism

### 3. Risk Scalars (regime mapping)
- NORMAL → 1.00
- ELEVATED → 0.85
- STRESS → 0.55
- CRISIS → 0.30

**Frozen:** Scalar values, EWMA smoothing (alpha=0.25), bounds [0.25, 1.0]

### 4. Application Convention
- 1-rebalance lag (apply scalar[t-1] to weights[t])
- Portfolio-level only (no sleeve-specific scaling)
- Missing scalar default: 1.0

**Frozen:** Lag convention, scaling level, default behavior

---

## What Changes Require v2

**The following changes are OUT OF SCOPE for v1 and require v2:**

### 1. Threshold Tuning (Stage 7)
- Adjusting regime enter/exit thresholds
- Changing stress condition definitions
- Modifying hysteresis parameters
- Tuning anti-thrash duration

**v1 Stance:** Thresholds are fixed. Stage 6.5 validation confirms they are "good enough."

**v2 Track:** Post-deployment, if stress detection is suboptimal, tune thresholds in v2 research track.

### 2. Sleeve-Level Scaling (Stage 6)
- Different scalars for different sleeves (e.g., more aggressive VRP de-risking)
- Sleeve-specific regime thresholds
- Differential stress response by engine type

**v1 Stance:** Portfolio-level only. Sleeve-level adds complexity.

**v2 Track:** After v1 validated, test sleeve-specific scaling in research track.

### 3. Convexity Overlays (Stage 8)
- Conditional VX calls during STRESS/CRISIS
- Options-based tail hedging
- Convexity budget allocation

**v1 Stance:** Always-on passive scaling only. No active overlays.

**v2 Track:** Integrate allocator-gated convexity after v1 stable.

### 4. Incremental State Computation (Stage 9)
- True online state updates (no rolling windows from scratch)
- Efficient warmup period handling
- Enables `mode="compute"` for production

**v1 Stance:** Use `mode="precomputed"` to sidestep warmup issues.

**v2 Track:** Implement incremental computation for live trading efficiency.

### 5. New State Features
- Additional volatility measures (VIX, MOVE)
- Liquidity proxies (bid-ask spreads, volume)
- Macro regime indicators (yield curve, credit spreads)
- Regime model outputs (HMM probabilities)

**v1 Stance:** 10 features are sufficient. Keep it simple.

**v2 Track:** Add features only if v1 misses critical stress signals.

### 6. Machine Learning or Optimization
- Learned regime classification
- Optimized thresholds or scalars
- Dynamic feature weighting
- Reinforcement learning for risk budgets

**v1 Stance:** Rule-based only. No black boxes.

**v2 Track:** ML may be introduced in v2+ if interpretability is preserved.

---

## Change Policy (Institutional Standard)

**v1 is FROZEN. Changes follow this process:**

### Allowed Without Version Bump (Patches)
- Bug fixes (e.g., NaN handling, edge cases)
- Documentation updates
- Artifact format improvements (backward-compatible)
- Performance optimizations (no logic changes)

### Requires v2 (Major Changes)
- Threshold adjustments
- New features or removal of features
- Regime logic changes
- Scalar mapping changes
- Sleeve-level scaling
- Convexity overlays
- `mode="compute"` production enablement

**v2 Development Process:**
1. Develop in parallel research track
2. Paper integration with v1 production copy
3. Two-pass audit (v1 baseline vs v2 scaled)
4. Stage 6.5 validation on v2
5. Formal promotion decision
6. Freeze v2, deploy alongside or replace v1

**No changes are injected directly into v1 production.**

---

## Deployment Checklist

**Before deploying v1 to production:**

- [ ] Run two-pass audit on canonical window (2020-2025)
- [ ] Complete Stage 6.5 validation (4 questions)
- [ ] Validate MaxDD reduction (2-5% improvement)
- [ ] Validate regime stickiness (<20 transitions/year)
- [ ] Validate stress flagging (2020 Q1, 2022 in top de-risk events)
- [ ] Sign-off on validation checklist
- [ ] Set `precomputed_run_id` to validated baseline
- [ ] Update production config: `enabled=true, mode="precomputed"`
- [ ] Deploy and monitor

**Once deployed, v1 is frozen. Future enhancements go through v2 track.**

---

## Monitoring & Maintenance (Post-Deployment)

**Weekly/Monthly:**
- Run two-pass audit with latest data
- Review regime transitions (thrashing check)
- Review top de-risk events (stress flagging check)
- Verify scalar distribution (mean should be 0.90-0.98)

**Quarterly:**
- Full Stage 6.5 re-validation
- Compare v1 results to baseline (allocator off)
- Decision: Keep v1, or promote v2 candidate

**Annually:**
- Review threshold stability across expanding window
- Assess if v2 development warranted (new features, tuning, sleeves)

**Alerts (Trigger v2 Research):**
- MaxDD increases vs baseline (allocator counterproductive)
- Regime thrashing (>30 transitions/year)
- Stress windows missed (2020/2022-style events not flagged)
- Mean scalar <0.85 (over-aggressive de-risking)

---

## Summary

**Allocator v1 is:**
- Rule-based, deterministic risk governor
- Portfolio-level exposure scaling (0.25–1.0)
- Production mode: `precomputed` (safe, auditable)
- Frozen at v1.0 (December 18, 2024)

**What's locked:**
- 10 state features
- 4 regime classification logic
- Risk scalar mapping
- 1-rebalance lag convention

**What requires v2:**
- Threshold tuning (Stage 7)
- Sleeve-level scaling (Stage 6)
- Convexity overlays (Stage 8)
- Incremental state (Stage 9)
- New features or ML

**Deployment:**
- Use `mode="precomputed"` with validated baseline
- Monitor weekly/monthly
- Freeze v1, develop v2 in parallel research track

**Allocator v1 is production-ready and frozen.**

---

**Freeze Authority:** Futures-Six Development Team  
**Freeze Date:** December 18, 2024  
**Next Review:** March 2025 (quarterly)  
**Version:** v1.0 (FROZEN)

---

**End of Document**

