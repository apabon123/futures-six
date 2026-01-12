# Futures-Six — System Construction & Architectural Decomposition

## 1. Purpose of This Document

Futures-Six is constructed as an institutional-grade systematic macro platform.

The system explicitly separates:

- **Economic return generation** (Engines / Meta-Sleeves)
- **Conditional risk control** (Allocator)

This document defines that separation.

**Its purpose is to:**

- Prevent logic leakage between layers
- Clarify where new ideas belong
- Ensure architectural consistency as the system evolves
- Make the system legible and auditable

**This document is conceptual, not operational.**

---

## 2. What an Engine Is

An **Engine** (Meta-Sleeve) is an unconditional economic hypothesis.

### Definition

An engine expresses an economic return source that operates continuously and does not decide when to turn itself off.

### Required Properties

An engine must be:

- **Always-on**
- **Unconditional**
- **Directional or relative** (but not state-aware)
- **Independent of portfolio context**
- **Portable across allocators**
- **Evaluated on unconditional behavior**

### What Engines Do

- Generate expected returns over time
- Accept losses as part of the economic hypothesis
- Remain active through all market conditions

### What Engines Do Not Do

Engines do not:

- Detect regimes
- Gate themselves
- Scale risk based on stress
- Activate crisis protection
- Respond to drawdowns

### Canonical Examples

- Trend
- Cross-Sectional Momentum
- Carry
- Volatility Risk Premium
- Curve Relative Value

---

## 3. What an Allocator Is

The **Allocator** is a risk control system, not an alpha engine.

### Definition

The allocator decides how much of the engines is allowed to express at any point in time.

### Core Responsibility

- Control exposure
- Control leverage
- Control risk under stress
- Preserve system survivability

### What the Allocator Does

- Observes market state
- Classifies risk regimes
- Transforms risk budgets
- Applies exposure constraints
- Activates defensive overlays

### What the Allocator Does Not Do

The allocator does not:

- Generate returns
- Predict markets
- Override engine direction
- Optimize Sharpe
- Learn directly from P&L

---

## 4. Canonical System Architecture (Frozen Conceptual Stack)

**Futures-Six is constructed as a layered, single-responsibility system.**

Each layer answers one and only one question, and no layer is allowed to subsume the responsibilities of another.

### The Canonical Execution Stack (Authoritative Order)

The canonical execution stack is:

1. **Engine Signals** (alpha)
2. **Engine Policy** (gates / throttles)
3. **Portfolio Construction** (static weights)
4. **Discretionary Overlay** (bounded tilts)
5. **Risk Targeting** (vol → leverage)
6. **Allocator** (risk brake)
7. **Margin & Execution Constraints**

**This ordering is authoritative. All future development must preserve these boundaries.**

---

### 1️⃣ Engine Signals (Alpha Generation)

**Purpose:** Generate directional or convex return streams.

**Allowed:**
- Signal construction
- Lookbacks
- Feature transforms
- Entry / exit logic
- Holding period definitions

**Not Allowed:**
- Risk scaling
- Volatility targeting
- Regime logic
- Portfolio awareness
- Discretionary overrides

**Examples:**
- Trend signals
- Cross-sectional momentum
- Carry
- VRP
- Intraday / micro-alpha (e.g., RSV, dealer flow)

**Key Principle:** Engines express belief, not permission.

---

### 2️⃣ Engine Policy (Validity & Selectivity)

**Purpose:** Determine whether and how much an engine should participate given context.

**Key Principle:** Engine Policy is a validity filter, not an optimizer.

**Allowed:**
- Binary gates (ON / OFF)
- Continuous throttles (0–100%)
- Contextual conditioning:
  - gamma imbalance
  - skew
  - dispersion
  - vol-of-vol
  - event risk (CPI, NFP, FOMC)
- Slow-moving, explainable rules

**Not Allowed:**
- Directional signals
- Sharpe optimization
- Fast PnL feedback
- Weight optimization
- Portfolio-level leverage control

**PnL Usage (Strictly Limited):**
- Allowed only for impairment detection
- Used asymmetrically (to reduce or disable, not to add)

**Engine Policy answers:** "Is this engine structurally valid in this environment?"

---

### 3️⃣ Portfolio Construction (Static Weights)

**Purpose:** Define the baseline composition of the system.

**Allowed:**
- Static engine weights
- Long-term conviction weights
- Transparent aggregation rules

**Not Allowed:**
- Dynamic optimization
- Regime-dependent reweighting
- PnL-driven allocation
- Allocator logic

**Key Principle:** These weights are base weights, later modified by policy, discretion, and risk controls.

---

### 4️⃣ Discretionary Overlay (Bounded Tilts)

**Purpose:** Provide a controlled outlet for high-level discretionary conviction without breaking systematic discipline.

**Placement:** After portfolio construction, before risk targeting.

**Allowed:**
- Bounded multipliers on:
  - asset classes
  - regions
  - sleeves / themes
- Slow frequency (weekly / rebalance)
- Explicit intent logging

**Not Allowed:**
- Signal overrides
- Individual trade control
- Bypassing policy or allocator
- Unbounded leverage changes

**Hierarchy Rule:** If an engine is gated OFF by policy, discretion cannot turn it back on. Discretion may tilt exposure, never bypass risk.

---

### 5️⃣ Risk Targeting (Volatility → Leverage)

**Purpose:** Define how large the portfolio is by design.

**Key Principle:** This layer encodes risk appetite, not risk control.

**Allowed:**
- Target portfolio volatility
- Equivalent leverage choice (e.g., 7×)
- Static or very slow updates

**Not Allowed:**
- Regime logic
- Stress detection
- Engine selection
- Dynamic brakes

**Key Principle:** This layer is always on and upstream of the allocator.

**Risk targeting answers:** "How big do I trade in normal conditions?"

---

### 6️⃣ Allocator (Risk Brake)

**Purpose:** Enforce survivability constraints during stress.

**Key Principle:** Allocator is a temporary brake, not a steering wheel.

**Allowed:**
- Portfolio-level scalars
- Coarse regimes (e.g., NORMAL / CRISIS)
- Stress & tail indicators:
  - drawdown cascades
  - correlation spikes
  - extreme volatility
- Rare intervention (especially in high-risk profiles)

**Not Allowed:**
- Engine-level weighting
- Signal awareness
- Alpha optimization
- Discretionary overrides

**Allocator Profiles:** Multiple allocator profiles may exist (H / M / L), differing only in risk tolerance, not architecture.

**Allocator answers:** "Is it safe to remain fully levered right now?"

---

### 7️⃣ Margin & Execution Constraints

**Purpose:** Enforce hard, external constraints.

**Allowed:**
- Margin checks
- Contract sizing limits
- Liquidity constraints
- Execution mechanics

**Not Allowed:**
- Strategy logic
- Risk decisions
- Feedback into allocator or policy

**Key Principle:** Margin is a post-construction constraint, not a decision signal.

---

### Candidate Variable Classification (Authoritative)

| Variable | Correct Layer |
|----------|---------------|
| Gamma imbalance | Engine Policy (primary), Allocator (tail only) |
| Skew | Engine Policy |
| Dispersion | Engine Policy |
| Vol-of-vol | Engine Policy |
| Event calendar | Engine Policy |
| Portfolio drawdown | Allocator |
| Cross-asset correlation | Allocator |
| RSV / order flow | Engine Signal |
| PnL (short-term) | ❌ Not allowed |
| PnL (impairment) | Engine Policy (defensive only) |

---

### Explicit Anti-Patterns (Do Not Implement)

- ❌ Allocator weighting sleeves
- ❌ Engine policy optimizing Sharpe
- ❌ Discretion bypassing gates or brakes
- ❌ Margin logic inside allocator
- ❌ Regime-dependent leverage inside risk targeting
- ❌ Fast feedback loops

**Violations of these rules break auditability and production safety.**

---

### Current System Status (As of January 2026)

- **Engines:** v1 COMPLETE
- **Engine Policy:** NOT YET IMPLEMENTED
- **Portfolio Construction:** Static v1
- **Discretionary Overlay:** Defined, optional
- **Risk Targeting:** ✅ **Phase 1C COMPLETE** (Production-Ready)
- **Allocator v1:** Institutional / Low-Risk Reference (Production-Ready)
- **Allocator v2 (H/M/L):** ✅ **Phase 1C COMPLETE** (H/M/L Profiles Production-Ready)

**Phase 1C Completion (January 2026):**
1. ✅ Risk Targeting layer implemented and validated
2. ✅ Allocator-H/M/L profiles implemented and validated
3. ✅ End-to-end integration validated (RT → Allocator application)
4. ✅ All artifacts auditable and deterministic
5. ✅ Contract tests prevent regressions

**Next development steps:**
1. Build Engine Policy v1 (Phase 2)
2. Paper-live deployment (Phase 3)

---

### Phase 1C: Risk Targeting + Allocator Integration (COMPLETE — January 2026)

**Status:** ✅ **COMPLETE** — Production-Ready

**Phase 1C Objectives:**
1. ✅ Implement Risk Targeting layer (Layer 5: vol → leverage)
2. ✅ Implement Allocator-H/M/L profiles (Layer 6: risk brake)
3. ✅ Prove end-to-end integration (RT → Allocator application)
4. ✅ Validate artifact integrity and deterministic output
5. ✅ Establish contract tests to prevent regressions

**Golden Proof Run (Phase 1C Acceptance Artifact):**

**Run ID:** `rt_alloc_h_apply_precomputed_2024`

**Config:**
```yaml
risk_targeting:
  enabled: true
  target_vol: 0.20
  leverage_cap: 7.0
  leverage_floor: 1.0

allocator_v1:
  enabled: true
  mode: "precomputed"  # Uses scalars from rt_alloc_h_apply_proof_2024
  precomputed_run_id: "rt_alloc_h_apply_proof_2024"
  profile: "H"
```

**Validator:** `scripts/diagnostics/validate_phase1c_completion.py rt_alloc_h_apply_precomputed_2024` must PASS

**Acceptance Criteria (All Passed):**
1. ✅ Allocator artifacts show active intervention (% active < 0.999: 42.3%, min scalar: 0.68)
2. ✅ RT + Alloc-H returns differ from RT-only (difference: 0.000944)
3. ✅ Weight scaling verified: `final_weights ≈ post_rt_weights * multiplier` (max error < 0.001)
4. ✅ ExecSim logs show: "Risk scalars applied: X/52 rebalances" where X > 0

**Proof Config Location:** `configs/proofs/phase1c_allocator_apply.yaml`

**Important Nuance (Documented for Future Reference):**

Phase 1C validation uses a **two-step process**:
1. **Step 1:** Compute allocator scalars in one run (`rt_alloc_h_apply_proof_2024` in `compute` mode)
2. **Step 2:** Apply scalars via `precomputed` mode in another run (`rt_alloc_h_apply_precomputed_2024`)

**This is acceptable for Phase 1C** because it proves:
- ✅ Allocator application path works correctly
- ✅ Config plumbing is correct
- ✅ Weight scaling is deterministic and auditable
- ✅ End-to-end integration is sound

**Behavioral Difference (To Be Validated in Phase 2/3):**

There is a difference between:
- **`compute` mode:** Compute-and-apply in-loop (live-like behavior)
- **`precomputed` mode:** Compute once, apply later (replay behavior)

**Phase 1C proves the application path and config plumbing.**  
**Phase 2/3 will validate compute-and-apply stability** (or explicitly choose `precomputed` for paper-live v0 if that's acceptable).

**Phase 1C Implementation Summary:**

**Risk Targeting Layer:**
- ✅ Leverage calculation: correct (target vol → leverage conversion)
- ✅ Weight scaling: correct (normalizes to unit gross, applies leverage)
- ✅ Artifacts: panel data fix (dedupe by `['date', 'instrument']`)
- ✅ Vol gap explanation: rebalance frequency effect (7.3% realized vs 20% target is expected)

**Allocator Profiles:**
- ✅ Profile-H: High risk tolerance (rare intervention, tail-only)
- ✅ Profile-M: Medium risk tolerance (balanced)
- ✅ Profile-L: Low risk tolerance (conservative, institutional-style)
- ✅ Contract tests: prevent regressions in regime scalar mappings

**Artifacts (All Validated):**
- ✅ `risk_targeting/leverage_series.csv` (time series)
- ✅ `risk_targeting/realized_vol.csv` (time series)
- ✅ `risk_targeting/weights_pre_risk_targeting.csv` (panel: date × instrument)
- ✅ `risk_targeting/weights_post_risk_targeting.csv` (panel: date × instrument)
- ✅ `risk_targeting/params.json` (once per run)
- ✅ `allocator/regime_series.csv` (time series)
- ✅ `allocator/multiplier_series.csv` (time series)
- ✅ `allocator_risk_v1_applied.csv` (rebalance-aligned)

**Contract Tests:**
- ✅ `tests/test_risk_targeting_contracts.py` — RT semantic correctness
- ✅ `tests/test_allocator_profile_activation.py` — Allocator profile correctness
- ✅ All tests pass, prevent regressions

**See Also:**
- `docs/PHASE_1C_FINAL_ANALYSIS.md` — Detailed analysis
- `docs/PHASE_1C_BUG_FIXES_COMPLETE.md` — Bug fixes summary
- `docs/PHASE_1C_PROOF_RUN.md` — Proof run documentation
- `docs/PHASE_1C_HANDOFF.md` — Status summary

---

### Final Canonical Statement

**Engines express belief.**  
**Policy enforces validity.**  
**Discretion expresses conviction.**  
**Risk targeting sizes the book.**  
**Allocator enforces survival.**

**This hierarchy is non-negotiable.**

---

## 5. Allocator Architectural Decomposition

The allocator is composed of four strictly ordered subsystems.

**A. State Estimation**  
**B. Regime Interpretation**  
**C. Risk Transformation**  
**D. Exposure Application**

Each subsystem has allowed and forbidden responsibilities.

### A. State Estimation

**"What does the world look like right now?"**

This layer measures observable conditions.

**Examples:**

- Realized volatility
- Volatility acceleration
- Cross-asset correlation
- Trend dispersion / breadth
- Liquidity or stress proxies

**Rules:**

- Descriptive only
- Continuous outputs
- No decisions
- No exposure changes

**Model Placement:**

- Regime models (e.g., HMMs) are permitted here only
- Outputs are treated as features, not commands

### B. Regime Interpretation

**"What risk state are we in?"**

This layer maps observations to risk regimes.

**Examples:**

- Normal
- Elevated risk
- Stress
- Crisis

**Rules:**

- Rule-based
- Sticky (hysteresis)
- Descriptive, not predictive
- Multiple inputs may vote

**Model Placement:**

- Regime model outputs may contribute probabilistically
- No single model has authority

### C. Risk Transformation

**"How should risk change given the regime?"**

This is the core allocator logic.

**Examples:**

- Volatility targeting
- Gross exposure scaling
- Sleeve risk compression
- Convexity budget decisions

**Rules:**

- Deterministic
- Monotonic (worse regime → less risk)
- Bounded
- Explainable

**🚫 No learning**  
**🚫 No regime inference**  
**🚫 No prediction**

### D. Exposure Application

**"How do we mechanically apply this decision?"**

This layer executes allocator decisions.

**Examples:**

- Scale sleeve weights
- Cap leverage
- Apply drawdown governors
- Activate convexity overlays

**Rules:**

- Mechanical only
- No logic discovery
- No state awareness

---

## 6. Common Misclassifications

This section defines frequent boundary errors.

### Ideas That Feel Like Engines but Are Allocator Logic

- "Only trade trend when volatility is low"
- "Turn off carry during stress"
- "Activate VX when crash probability rises"

**These are allocator decisions.**

### Ideas That Feel Like Allocator Logic but Are Engines

- "Short volatility when VIX > realized volatility"
- "Trend signal scaled by its own volatility"
- "Carry conditioned on curve shape"

**These are engines if unconditional.**

### Rule of Thumb

- If an idea **decides when risk should be taken**, it belongs in the allocator.
- If an idea **earns returns regardless of regime**, it is an engine.

---

## 7. Lifecycle: How Ideas Move Through the System

1. Engines are researched and validated in isolation
2. Engines are integrated into the baseline portfolio
3. Allocator logic is layered on top
4. Production logic is frozen
5. New ideas enter via a parallel research track
6. Promotions are versioned and explicit

**Engines and allocator logic evolve independently.**

---

## 8. Why This Separation Matters

This architecture:

- Prevents hidden timing logic
- Preserves engine portability
- Enables safe leverage
- Supports institutional governance
- Makes failures explainable
- Scales from small to large capital

**Futures-Six prioritizes survivability and clarity over cleverness.**

---

## 9. Allocator v1 Implementation (Stages 4A-5.5)

**Status:** Production-ready (December 2024)

### Overview

Allocator v1 is the first production implementation of the allocator architecture defined above. It follows the four-layer decomposition exactly:

**A. State Estimation** → `AllocatorStateV1` (10 features)  
**B. Regime Interpretation** → `RegimeClassifierV1` (4 regimes)  
**C. Risk Transformation** → `RiskTransformerV1` (risk scalars)  
**D. Exposure Application** → ExecSim integration (weight scaling)

### Implementation Layers

#### A. State Estimation (`AllocatorStateV1`)

**Purpose:** Measure observable portfolio and market conditions

**10 Canonical Features:**
- **Vol / Acceleration:** `port_rvol_20d`, `port_rvol_60d`, `vol_accel`
- **Drawdown / Path:** `dd_level`, `dd_slope_10d`
- **Cross-Asset Correlation:** `corr_20d`, `corr_60d`, `corr_shock`
- **Engine Health:** `trend_breadth_20d`, `sleeve_concentration_60d`

**Properties:**
- Descriptive only (no decisions)
- Continuous outputs (daily)
- Deterministic computation
- 8 required + 2 optional features

**Artifact:** `allocator_state_v1.csv` (saved with every run)

#### B. Regime Interpretation (`RegimeClassifierV1`)

**Purpose:** Map state to discrete risk regimes

**4 Risk Regimes:**
- **NORMAL:** Typical market conditions
- **ELEVATED:** Increased volatility or correlation
- **STRESS:** Significant drawdown or volatility spike  
- **CRISIS:** Extreme conditions requiring defensive positioning

**Logic:**
- Rule-based classification (no ML)
- Uses 4 stress condition signals (vol acceleration, correlation shock, drawdown depth, drawdown slope)
- Hysteresis (separate enter/exit thresholds)
- Anti-thrash (minimum 5 days in regime)
- Sticky by design (prevents regime flapping)

**Artifact:** `allocator_regime_v1.csv` (daily regime series)

#### C. Risk Transformation (`RiskTransformerV1`)

**Purpose:** Convert regime to portfolio-level risk scalar

**Canonical Mapping:**
- NORMAL → 1.00 (no adjustment)
- ELEVATED → 0.85 (15% reduction)
- STRESS → 0.55 (45% reduction)
- CRISIS → 0.30 (70% reduction)

**Properties:**
- Deterministic mapping
- Monotonic (worse regime → lower scalar)
- Bounded [0.25, 1.0]
- EWMA smoothed (alpha=0.25, ~5d half-life)

**Artifact:** `allocator_risk_v1.csv` (daily risk scalars)

#### D. Exposure Application (ExecSim Integration)

**Purpose:** Apply risk scalars to portfolio weights

**Implementation Modes:**
1. **`mode: "off"`** - Compute artifacts only, no weight scaling (default)
2. **`mode: "precomputed"`** - Load scalars from prior run, apply with lag (Stage 5.5)
3. **`mode: "compute"`** - On-the-fly computation (has warmup issues, not recommended)

**Application Convention:**
- 1-rebalance lag: `risk_scalar[t-1]` applied to `weights[t]`
- No lookahead bias
- Preserves engine direction (scales uniformly)

**Artifacts:** `weights_raw.csv`, `weights_scaled.csv`, `allocator_risk_v1_applied_used.csv`

### Key Design Principles

**1. Separation of Concerns**
- Each layer has a single responsibility
- No cross-layer logic leakage
- Testable in isolation

**2. Determinism**
- No ML or optimization (rule-based only)
- Fully reproducible
- Auditable at every step

**3. Stickiness**
- Hysteresis prevents thrashing
- Anti-thrash rules enforce minimum regime duration
- Smoothing prevents jerk

**4. Artifact-First**
- All computations saved automatically
- Complete audit trail
- Offline analysis without re-running backtests

**5. Mode Flexibility**
- Can run in "off" mode for research
- Two-pass audit for validation
- Precomputed mode for production

### Two-Pass Audit Framework (Stage 5.5)

**Purpose:** Validate allocator impact before live deployment

**Workflow:**
1. **Pass 1 (Baseline):** Run with `allocator_v1.enabled=false`
   - Generates portfolio returns
   - Computes state/regime/risk artifacts
   - Saves `allocator_risk_v1_applied.csv`

2. **Pass 2 (Scaled):** Run with `mode="precomputed"`
   - Loads scalars from Pass 1
   - Applies them with proper lag
   - Generates scaled portfolio

3. **Comparison Report:**
   - CAGR, vol, Sharpe, MaxDD
   - Worst month/quarter
   - Scalar usage statistics

**Scripts:**
- `scripts/diagnostics/run_allocator_two_pass.py` (orchestration)
- `scripts/diagnostics/compare_two_runs.py` (comparison report)

### Configuration

In `configs/strategies.yaml`:

```yaml
allocator_v1:
  enabled: false              # Master switch
  mode: "off"                 # "off" | "compute" | "precomputed"
  precomputed_run_id: null    # Required if mode="precomputed"
  precomputed_scalar_filename: "allocator_risk_v1_applied.csv"
  apply_missing_scalar_as: 1.0
  state_version: "v1.0"
  regime_version: "v1.0"
  risk_version: "v1.0"
```

### Validation Status

**Acceptance Criteria Met:**
- ✅ All 10 state features computed correctly
- ✅ Regime classification is sticky (no thrashing)
- ✅ Risk scalars are bounded and deterministic
- ✅ Two-pass audit produces different results (scalars applied)
- ✅ Complete artifact trail
- ✅ Comparison reports generated

**Production-Ready:** December 2024

### Future Enhancements

**Stage 6:** Sleeve-level risk scalars (differential scaling by engine type)  
**Stage 7:** Threshold tuning against historical stress events  
**Stage 8:** Convexity overlays (VIX calls) gated by regime  
**Stage 9:** True incremental state computation (resolve warmup period)

### References

- **Implementation Docs:** `docs/ALLOCATOR_V1_STAGE_4_COMPLETE.md`, `docs/ALLOCATOR_V1_QUICK_START.md`
- **Source Code:** `src/allocator/` (state_v1, regime_v1, risk_v1, scalar_loader)
- **Diagnostics:** `scripts/diagnostics/run_allocator_*.py`
- **Tests:** Two-pass audit validates allocator reduces MaxDD without destroying returns

---

End of Document

