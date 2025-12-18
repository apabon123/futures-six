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

## 4. Allocator Architectural Decomposition

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

## 5. Common Misclassifications

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

## 6. Lifecycle: How Ideas Move Through the System

1. Engines are researched and validated in isolation
2. Engines are integrated into the baseline portfolio
3. Allocator logic is layered on top
4. Production logic is frozen
5. New ideas enter via a parallel research track
6. Promotions are versioned and explicit

**Engines and allocator logic evolve independently.**

---

## 7. Why This Separation Matters

This architecture:

- Prevents hidden timing logic
- Preserves engine portability
- Enables safe leverage
- Supports institutional governance
- Makes failures explainable
- Scales from small to large capital

**Futures-Six prioritizes survivability and clarity over cleverness.**

---

End of Document

