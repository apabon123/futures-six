# Phase 4 Engine Research Protocol — Anti-Snooping Checklist

**Research Governance Document** (Not an SOT)

**Last updated: January 2026**

---

## Purpose

This checklist governs all Phase 4 engine-quality work. Its sole objective is to prevent in-sample overfitting, data snooping, and silent optimization while improving belief quality under a frozen architecture.

**This document is a research contract, not an SOT. It constrains how engines may be studied and modified.**

---

## A. Research Law (Hard Constraints)

These are non-negotiable.

### 1. Only Engines May Change

- Construction, Risk Targeting, Allocator, Diagnostics are **frozen**.
- No "temporary" plumbing changes to help an engine.

### 2. No Grid Search

- No parameter sweeps over horizons, skips, weights, clips, thresholds.
- No selecting the best of many variants.

### 3. Hypothesis First, Code Second

Every variant must be justified **before running** by:
- A known literature effect, **or**
- A structural failure observed in diagnosis.

### 4. Single-Change Rule

- Each experiment may introduce **one conceptual change only**.
- No compound tuning.

### 5. Phase 2 is the Only Promotion Authority

- Phase 3B attribution is monitoring only.
- No automatic promotion or demotion from Phase 3B.

### 6. Evaluation Target is Post-Construction

- Standalone improvement is **insufficient**.
- The objective is improved **Post-Construction portfolio behavior**.

---

## B. Legal Research Workflow (Must Follow This Order)

For every engine experiment:

### Step 1 — Belief Autopsy (No Code Changes)

**Purpose**: Diagnose the current promoted engine.

**Allowed outputs**:
- Standalone engine metrics (full window + eval window)
- Yearly returns table
- Drawdown path, time-under-water
- Contribution by asset
- Rank stability / turnover
- Post-Construction attribution slices

**Forbidden**:
- No parameter changes
- No conditional slicing to justify future tuning

**Deliverable**:
> A short diagnostic memo: "Observed failure modes of current engine."

---

### Step 2 — Hypothesis Specification

Before coding, write:

| Field | Description |
|-------|-------------|
| Variant name | Descriptive identifier |
| Single conceptual change | One change only |
| Theoretical justification | Literature or structural failure |
| Expected failure mode addressed | What this fixes |
| Expected risk if hypothesis is wrong | What could go wrong |

**If this cannot be written, the experiment is illegal.**

---

### Step 3 — Phase 0 / Phase 1 Standalone Test

**Purpose**: Check signal sanity and implementation correctness.

**Requirements**:
- Same universe
- Same window
- No tuning based on results

**Reject if**:
- Sign-only Sharpe < 0.2
- Obvious instability or pathologies

---

### Step 4 — Phase 2 Integrated Test (Promotion Gate)

**Purpose**: Test in full frozen system.

**Rules**:
- Only this engine variant changes
- Same canonical window
- Same diagnostics harness

**Promotion criteria**:
- Improves or stabilizes Post-Construction Sharpe
- Does not materially worsen drawdown or regime fragility
- Does not increase correlation spikes with Trend

**If Phase 2 fails → variant is rejected permanently.**

---

### Step 5 — Phase 3B Monitoring (Stewardship Only)

**Purpose**: Understand system-level impact.

**Rules**:
- No tuning based on Phase 3B
- No automatic gating or reweighting
- Attribution is for human understanding only

---

## C. Anti-Snooping Safeguards

These are explicitly designed to reduce false discovery:

| Safeguard | Description |
|-----------|-------------|
| Max 2 variants per engine per research cycle | Prevents parameter fishing |
| No trying multiple skip lengths / weights | Single hypothesis per experiment |
| No subperiod cherry-picking | No justifying changes with specific windows |
| No regime-conditioned rules in Phase 4 | Regime logic is frozen |
| All rejected variants recorded | Never retried without new evidence |

---

## D. Success Definition for an Engine

An engine is improved **only if**:

1. Post-Construction contribution improves materially
2. Multi-year bleed periods are reduced
3. Diversification vs other sleeves improves
4. Behavior is more stable across regimes

**Standalone Sharpe improvement alone is not sufficient.**

---

## E. Record-Keeping (Mandatory)

For every experiment, record:

| Field | Description |
|-------|-------------|
| Baseline run ID | Reference baseline |
| Variant run ID | Experiment run |
| Hypothesis statement | From Step 2 |
| Phase 1 summary | Standalone results |
| Phase 2 summary | Integration results |
| Final decision | Promoted / Rejected |

This creates a permanent research lineage.

---

## F. Summary Principle

> **Phase 4 is not optimization.**
>
> It is hypothesis-driven belief hygiene under frozen governance.
>
> **If any step feels like "tuning to the backtest," the experiment is illegal.**

---

## Related Documents

- **ROADMAP.md** § "Phase 4 Entry Criteria" — Entry conditions for Phase 4
- **PROCEDURES.md** § "Phase 3B vs Phase 2 Decision Authority" — Decision hierarchy
- **SYSTEM_CONSTRUCTION.md** § "Portfolio Construction v1" — Frozen construction spec
- **DIAGNOSTICS.md** § "Post-Construction Attribution" — Attribution tooling
