# ROADMAP.md — Futures-Six: Strategic Development Roadmap (2026–2028)

**Source-of-Truth Document**

**Last updated: Feb 2026**

---

## 1. Purpose of This Document

**ROADMAP.md** is a strategic, forward-looking, non-code source of truth that governs:

- Long-term development sequencing
- Meta-sleeve expansion plans
- Production deployment planning
- Research targets (Sharpe, diversification, crisis behavior)
- Iterative improvement cycles after initial production release
- High-level architecture constraints and priorities

This document complements:

| Document | Purpose |
|----------|---------|
| STRATEGY.md | What the system is |
| PROCEDURES.md | How the system is built |
| DIAGNOSTICS.md | How the system is validated |
| ROADMAP.md | What the system will become |

**This is the single source of truth for sleeve status, priorities, and development sequencing.**

## Related Documents

- [docs/SOTs/STRATEGY.md](docs/SOTs/STRATEGY.md): Sleeve definitions, signal specifications, position geometry
- [docs/SOTs/SYSTEM_CONSTRUCTION.md](docs/SOTs/SYSTEM_CONSTRUCTION.md): Architecture, layers, data flow
- [docs/SOTs/PROCEDURES.md](docs/SOTs/PROCEDURES.md): Promotion workflow, run creation, required artifacts
- [docs/SOTs/DIAGNOSTICS.md](docs/SOTs/DIAGNOSTICS.md): Required diagnostics and pass/fail criteria

---

## 2. Strategic Objective (2026–2028)

Futures-Six targets institutional CTA/Macro performance with:

- **Sharpe 1.5–2.0 unlevered**
  - + Safe portfolio-level leverage (allocator applied)
- Low correlation sleeves
- Diversified economic themes
- 100% deterministic, reproducible engineering

**The FIRST milestone is a production-ready system, not maximized Sharpe.**

Sharpe improvement happens in Post-Production Expansion Cycles.

**Explicit Statement**: Futures-Six intentionally launches with a minimum viable but architecturally complete system. Additional sleeves and refinements are accretive and governed post-launch. The initial production release (Core v9) represents a complete, institutionally realistic CTA/Macro architecture, not a maximally optimized system.

---

## 3. Current State (as of Core v9)

**Completed Meta-Sleeves:**

- **Trend Meta-Sleeve** — COMPLETE
- **CSMOM Meta-Sleeve** — COMPLETE
- **VRP Meta-Sleeve** (ongoing):
  - VRP-Core — PROMOTED
  - VRP-Convergence — PROMOTED
  - VRP-Alt — PROMOTED (Dec 2025)
  - Additional VRP sleeves still required for stability and diversification.

**Baseline:**

- Core v9 (Trend 52.4% + CSMOM 21.85% + VRP-Core 6.555% + VRP-Convergence 2.185% + VRP-Alt 13.11% + VX Carry 4.6% + Curve RV 8%)

**Performance (Canonical Window: 2020-01-06 to 2025-10-31):**

- **Core v8**: Sharpe 0.5820, CAGR 6.81%, MaxDD -17.13%, Vol 12.70%
- **Core v9**: Sharpe 0.6605, CAGR 9.35%, MaxDD -15.32%, Vol 12.01%
- **Improvement**: Sharpe +0.0785, CAGR +2.54%, MaxDD +1.81%, Vol -0.69%

*Note: All canonical metrics computed on canonical evaluation window (2020-01-06 to 2025-10-31). See `configs/canonical_window.yaml` for the authoritative window definition. See STRATEGY.md § "Baseline Evolution Summary" for full Core v3-v9 comparison table.*

### Phase-3 Status Note

As of Phase-3A, Futures-Six maintains a formal set of pinned baseline runs used for:

- Attribution
- Ablation testing
- Regression tracking

Performance improvements beyond this point are evaluated only relative to pinned baselines, ensuring controlled, auditable progress.

#### Phase 3A Status (Jan 2026) -> COMPLETE
- [x] **Policy Feature Plumbing**
    - [x] Extract `gamma_stress_proxy`, `vx_backwardation`, `vrp_stress_proxy` to `meta.json`
    - [x] Ensure policy features are purely data-driven (no config hardcoding)
- [x] **Governed Baseline Re-Freeze**
    - [x] Re-run canonical baseline with full feature tracking
    - [x] Pin new baseline: `canonical_frozen_stack_precomputed_phase3a_governed_...`
- [x] **RT & Allocator Governance**
    - [x] Fix RT telemetry (finite stats in meta.json)
    - [x] Formalize "Effective vs Inert" contract
- [x] **Phase 3A Statistical Baseline — Fully Governed**
    - [x] Established pinned baseline: `phase3a_statistical_baseline_governed_20251031`
    - [x] Effective/Eval start: 2020-03-20
    - [x] Implemented dual metrics (`returns span` vs `eval window`)
    - [x] Implemented strict precomputed lineage validation

**🔜 Next Steps (Calibration Sprint):**
- RT + Allocator calibration (distribution targets)
- Median gross 4.0–4.5×
- P90 gross 5.5–6.5×
- Cap rarely binds

**⏸ Defer:**
- Ablations until RT/Allocator behavior is frozen.
- [ ] **Next: Phase 3A Attribution Ablations** (Policy / RT / Allocator / Sleeves)

### Phase-4 / V1 hardening (Feb 2026)

- **VX calendar carry** → accepted into V1 baseline.
- **SR3 calendar spread carry** → implemented and validated in Phase-4 but deferred to post-V1 (minimize trading surface for initial release).
- **VRP metasleeve** → deferred to V2 pending formal engine policy (policy-conditioned engine).
- **SR3 curve RV** → remains part of V1 but flagged for Phase-4 diagnostic follow-up (Phase-2 vs Phase-4 atomic return stream comparison due to negative Phase-4 contribution).

### Core v7 Upgrade — VRP-Alt Integration (15%)

**Completed**: Phase-0, Phase-1, Phase-2, and scaling analysis for VRP-Alt.

**VRP-Alt promoted at 15% weight** into VRP Meta-Sleeve.

**Core v7 now defined as:**
- Trend: 60%
- CSMOM: 25%
- VRP-Core: 7.5%
- VRP-Convergence: 2.5%
- VRP-Alt: 15%

VRP-Convergence remains deprioritized but still present.

**Next action**: Continue VRP Meta-Sleeve diversification with additional VRP sleeves.

---

## 4. Short-Term Roadmap (Next 6–9 Months)

**Goal: Complete VRP Meta-Sleeve + Implement Carry Meta-Sleeve + Add Crisis Meta-Sleeve (Minimal) + Deploy First Production Version**

**Pre-Phase-B Meta-Sleeve Set:**
- Trend ✅
- CSMOM ✅
- VRP ✅
- Carry 🚧 (IN PROGRESS)
- Curve RV ✅ (Phase-2 Complete, PROMOTED to Core v9)
- Crisis 🚧 (Phase-0 IN PROGRESS — Minimal, defensive-only)

This establishes a complete, institutionally realistic CTA/Macro architecture prior to allocator design and Phase-B optimization.

### 4.1 Complete VRP Meta-Sleeve

**VRP Meta-Sleeve Status:**
- ✅ VRP-Core: Complete (Phase-2 integrated) — **First canonical VRP sleeve**
- ✅ VRP-Alt: Complete (Phase-2 integrated, promoted Dec 2025) — **Second canonical VRP sleeve**
- ⏸️ VRP-Convergence: Economically parked / deprioritized (2.5% weight, kept for historical continuity)
- ❌ VRP-FrontSpread (directional): **PARKED — Phase-0 FAIL**
- ❌ VRP-Convexity (VVIX threshold): **PARKED — Phase-0 FAIL** (threshold short VX1)
- ❌ VRP-RollYield: PARKED (Borderline Phase-0 result)
- ⏳ VRP-CalendarDecay: Future (Medium Priority)
- ❌ VRP-TermStructure: PARKED (Phase-0 failure)

**Target v1 VRP Meta-Sleeve Composition:**
- VRP-Core (required)
- VRP-Convergence (required)
- (Optional) VRP-CalendarDecay

**Governance note (2026-Q1 update):** Although VRP atomic sleeves were previously promoted into Core v7–v9, all new VRP research and refactoring is now evaluated under the Phase-4 engine-quality workflow. New VRP variants and refactors must first pass engine-quality runs (control + VRP only), and only afterwards be evaluated in integration runs against Core v9.

---

## Remaining VRP Sleeves That MUST Exist for a Serious VRP Meta-Sleeve

Below is the canonical list that institutional volatility premia research converges on. This is how you fill the VRP Meta-Sleeve with 4–6 sleeves.

### 🔵 VRP Sleeve #1 — VRP-Core (✓ COMPLETE)

Your only validated signal so far. Uses VIX − RV21 spread.

---

### 🔵 VRP Sleeve #2 — VRP-Alt (VRP-Richness, ALT-VRP) ✅ COMPLETE

**Description:** VIX vs RV5; short VX1 when VIX > RV5 (sensitive to spike decay). See [docs/SOTs/STRATEGY.md](docs/SOTs/STRATEGY.md) § "VRP-Alt" for signal and geometry.

**Stage:** Production (Core v7+ at 15% weight)

**Next step:** Maintain; no active changes.

---

### 🔵 VRP Sleeve #3 — VRP-Convexity Premium (VVIX Relative Value)

**Description:** VVIX threshold short VX1. See [docs/SOTs/STRATEGY.md](docs/SOTs/STRATEGY.md) § "VRP-Convexity".

**Stage:** Parked (Phase-0 fail)

**Next step:** Revisit only as conditioning feature or spread-style trade.

---

### 🔵 VRP Sleeve #3 — VRP-FrontSpread (Directional) ❌ PARKED

**Description:** VX1 − VX2 richness spread; short VX1 when VX1 > VX2. See [docs/SOTs/STRATEGY.md](docs/SOTs/STRATEGY.md) § "VRP-FrontSpread".

**Stage:** Parked (Phase-0 fail)

**Next step:** Revisit only as calendar-spread trade or regime input.

---

### 🔵 VRP Sleeve #4 — VRP-Structural (RV252) ❌ PARKED

**Description:** VIX vs RV252; short VX when VIX > RV252. See [docs/SOTs/STRATEGY.md](docs/SOTs/STRATEGY.md) § "VRP-Structural".

**Stage:** Parked (Phase-0 fail)

**Next step:** Revisit only as conditioning feature or different instrument.

---

### 🔵 VRP Sleeve #4 — VRP-Mid (RV126) ❌ PARKED

**Description:** VIX vs RV126; short VX when VIX > RV126. See [docs/SOTs/STRATEGY.md](docs/SOTs/STRATEGY.md) § "VRP-Mid".

**Stage:** Parked (Phase-0 fail)

**Next step:** Revisit only as conditioning feature or different instrument.

---

### 🔵 VRP Sleeve #5 — VRP-CalendarDecay (Tenor Ladder Decay)

This is closer to how actual volatility desks trade VRP.

**Idea:**

Each VX future decays toward its "anchor":
- VX1 → VIX
- VX2 → VX1
- VX3 → VX2

**Phase-0:**

- Short VX2 if VX2 > VX1
- Short VX3 if VX3 > VX2

Or basket-weighted (tenor decay sleeve).

This is more robust than RollYield, which relied on VX1 alone.

**Status:** Not yet implemented. Medium priority for Phase-0 testing.

---

### 🔵 VRP Sleeve #6 — VIX Curve Curvature (Fly) — PLANNED (v2, Low Priority)

**Instruments**: VX1, VX2, VX3

**Expression:**

Calendar fly (e.g., `2*VX2 − VX1 − VX3`)

**Motivation:**

Test whether VIX term-structure curvature contains incremental information beyond VX2–VX1 carry.

**Notes:**

- Liquidity beyond VX2 is limited
- Expected redundancy with VX calendar carry
- Included for research completeness; promotion unlikely unless behavior is clearly distinct

**Status:**

- Not tested in v1
- Deferred to post-v1 expansion cycle (v2)
- Low priority relative to other VRP sleeves

---

### 🔵 VRP Sleeve #7 — VRP-ReverseConvergence (Crisis Turning Point Sleeve)

This is a long-vol sleeve, which is needed for stabilization.

**Idea:**

When backwardation gets extreme, VRP carry unwinds violently.

$$VIX - VX1 > 3 \Rightarrow go\ long\ VX1$$

This can be:
- A Crisis Meta-Sleeve component
- OR a VRP shock absorber sleeve
- OR an exposure dial for VRP-Core

Phase-0 should show positive Sharpe because a long-vol sleeve during extreme backwardation often works.

**Status:** Not yet implemented. Lower priority (may belong in Crisis Meta-Sleeve).

---

### ❌ PARKED Sleeves

**VRP-RollYield (Status: PARKED — Borderline Phase-0)**

VRP-RollYield was intended to capture front-month roll-down carry by shorting VX1 when the future traded above VIX on a per-day-to-expiry basis. A Phase-0 sign-only implementation:

Short VX1 when:

$$roll\_yield = \frac{VX1 - VIX}{days\_to\_expiry} > 0$$

Phase-0 results over the canonical 2020–2025 window:

- Sharpe +0.02 (below the ≥ 0.10 Phase-0 bar)
- MaxDD –85.65% (slightly worse than the catastrophic threshold)
- Non-degenerate signal (short ~75% of days)

The diagnostics indicate a borderline, near-zero economic edge with an unacceptable path profile. Unlike VRP-Core and VRP-Convergence, the raw roll-yield sign rule does not meet the standard for promotion.

**Decision:** VRP-RollYield in this simple form is PARKED. The roadmap treats roll-down carry as a research topic for a future variant, potentially as a multi-tenor basket or integrated component.

**VRP-TermStructure (slope-only directional trades)**

**Status:** PARKED

**Phase-0 Results:**
- Confirmed via Phase-0: VX2 − VX1 slope is not a directional VRP signal
- Does not map cleanly to outright front-month selling
- High overlap with VIX–VX1 but weaker economics
- Better used as regime feature → not return engine

**Decision:** We PARK it here. The roadmap will revisit term-structure dynamics in one of the following forms:
- As a regime filter for VRP-Core, VRP-Convergence, or other VRP sleeves
- As part of a future Crisis / Long-Vol Meta-Sleeve
- As a candidate for calendar-spread-based sleeves rather than outright VX1 directionality

---

#### Final Expected VRP Meta-Sleeve Composition

For a clean, institutional, stable VRP Meta-Sleeve, the ideal set is:

| Sleeve | Status | Priority | Rationale |
|--------|--------|----------|-----------|
| VRP-Core | ✅ Complete | Required | **First canonical VRP sleeve** |
| VRP-Alt (RV5) | ✅ Complete | Required | **Second canonical VRP sleeve** (promoted Dec 2025), well-supported, diversifies VRP-Core |
| VRP-Convergence | ⏸️ Parked | Deprioritized | Economically parked (2.5% weight, kept for historical continuity only) |
| VRP-FrontSpread (directional) | ❌ PARKED | N/A | **Phase-0 FAIL** — Calendar-richness does not map to profitable outright VX1 short |
| VRP-Convexity Premium (VVIX threshold) | ❌ PARKED | N/A | **Phase-0 FAIL** — VVIX data available; threshold expression failed |
| VRP-Structural (RV252) | ❌ PARKED | N/A | **Phase-0 FAIL** (VX1/VX2/VX3) — Not an engine in simple form; revisit only as conditioning or different instrument/expression |
| VRP-Mid (RV126) | ❌ PARKED | N/A | **Phase-0 FAIL** (VX2/VX3) — Not an engine in simple form; revisit only as conditioning or different instrument/expression |
| VRP-CalendarDecay | ⏳ Future | Medium | Tenor-specific carry |
| VRP-ReverseConvergence | ⏳ Future | Low | Possibly Crisis Meta-Sleeve |
| VRP-RollYield | ❌ PARKED | N/A | Borderline Phase-0 result |
| VRP-TermStructure | ❌ PARKED | N/A | Phase-0 failure |

**Target: 4–6 sleeves in v1 VRP Meta-Sleeve:**
1. VRP-Core ✅ (First canonical VRP sleeve)
2. VRP-Alt (RV5) ✅ (Second canonical VRP sleeve, promoted Dec 2025)
3. (Optional) VRP-CalendarDecay ⏳ (Future)
4. (Optional) Additional VRP sleeves ⏳ (Future)

**Note**: VRP-Convergence is economically parked (2.5% weight) and not counted toward the v1 target composition.

### 4.1.3 VRP Meta-Sleeve Assembly (Post 4-Sleeve Completion)

A proper meta-sleeve requires:

- Sleeve-level return covariance modeling
- ERC or Max-Diversification weighting
- Stability filters (regime filters optional)

**Diagnostics:**

- Diversification benefit
- Crisis behavior
- VRP sleeve correlations
- Sleeve-level turnover

### 4.2 Build Carry Meta-Sleeve (IN PROGRESS)

**Status:** 🚧 IN PROGRESS

Carry Meta-Sleeve is chosen because:

- Carry sleeves have low correlation with Trend & VRP
- Carry is a major component of major CTAs
- Carry improves long-term Sharpe in a diversified system

**Completed / Promoted:**

- ✅ **SR3 Calendar Carry** — **PROMOTED** (Dec 2025)
  - **Phase-0**: PASS (Sharpe 0.6384, R2-R1 canonical pair)
  - **Phase-1**: PASS (Implementation complete, execution rules frozen)
  - **Phase-2**: PASS (Portfolio integration: MaxDD improvement +0.80%, correlation 0.04, Sharpe preserved)
  - **Role**: Risk stabilizer / diversifier (5% research weight)
  - **See**: `docs/SOTs/DIAGNOSTICS.md` § "SR3 Calendar Carry" for full development history and promotion decision

**Phase-0 Status (Dec 2025):**

- ✅ **VX Calendar Carry** — **COMPLETE / PROMOTED** (Dec 2025)
  - **Canonical Atomic Sleeve**: VX2–VX1_short (promoted to Core v8 baseline)
  - **Secondary Atomic Sleeve**: VX3–VX2_short (Phase-2 PASS, validated backup, non-default)
  - **Phase-0**: PASS (both short-spread variants show strong standalone carry expectancy)
  - **Phase-1**: PASS (z-scoring, vol targeting, execution rules frozen; two atomic sleeves)
  - **Phase-2**: PASS (Portfolio integration: Sharpe +0.0377, MaxDD +1.02%, Vol -0.71%, correlation -0.0508)
  - **Status**: PROMOTED — Canonical atomic sleeve integrated into Core v8 baseline (5% research weight)
  - **Promotion Rationale**: VX2–VX1_short shows slightly stronger Phase-2 improvements and liquidity advantages. VX3–VX2_short retained as validated secondary option with excellent diversification.
  - **Note**: Secondary variant retained as fallback for future trading reality considerations
  - **See**: `docs/SOTs/DIAGNOSTICS.md` § "VX Calendar Carry" for full development history and promotion decision

- ❌ **FX/Commodity Carry**: Parked for redesign
  - **Phase-0 Sanity Check Result**: Negative Sharpe (-0.69) across all assets (2020-2025)
  - **Findings**: Sign-only roll yield strategy showed negative alpha in recent years
  - **Status**: Remains on roadmap for redesign (e.g., sector-based roll yield, DV01-neutral carry, regime-dependent filters)

**Planned Atomic Sleeves (first wave)**

- Futures Basis Carry (e.g., ES, TY, CL, NG, GC)
- Yield Curve Carry
- Roll Down/Carry on Vol (distinct from VRP roll yield)
- Cross-Sectional Carry (FX-like logic applied to commodities/bonds)

We will likely build 3–4 sleeves for v1.

**Goal for Carry v1**

- Add 0.15–0.25 Sharpe to the system
- Improve diversification
- Contribute positively during low-volatility equity regimes

### 4.3 Build Curve Relative Value (Curve RV) Meta-Sleeve

**Status:** v1 COMPLETE (Core v9); v2 PLANNED (Post-v1)

**Description:** SR3 curve momentum (Rank Fly 5%, Pack Slope 3%); market-neutral spreads and flies. See [docs/SOTs/STRATEGY.md](docs/SOTs/STRATEGY.md) for signal definitions.

**Stage:** Production (Core v9)

**Next step:** v2 Treasury Curve RV (deferred).

#### v2 (PLANNED) — Treasury Curve RV

**Instruments**: ZT (2-year) / ZF (5-year) / ZN (10-year) / UB (30-year)

**Planned Expressions:**
- 2s10s slope momentum
- 5s30s slope momentum
- 2s5s10s fly momentum
- 5s10s30s fly momentum

**Requirements:**
- DV01-aware construction (rates contracts have different DV01s)
- Phase-0 to be conducted post-v1 freeze

**Status**: Deferred to v2 expansion cycle

#### Deferred / Separate Sleeves

**Cross-Market Basis RV (SOFR ↔ Treasuries):**
- Macro RV, not pure Curve RV
- Deferred to v2+

**VIX Curve Shape:**
- Treated under VRP / Volatility sleeves
- Explicitly excluded from Curve RV

**Architectural Guardrail:** Curve RV is a distinct economic engine and will not be blended or hybridized with Carry or Crisis during Phase-0/1.

### 4.4 Build Crisis / Tail Risk Meta-Sleeve (v1 COMPLETE — NO PROMOTION)

**Status:** ✅ v1 COMPLETE — NO PROMOTION

**Summary:**
- Phase-0/1/2 completed
- VX3 failed Phase-2 due to 2020 Q1 fast-crash window
- Always-on convexity not adopted in v1

**Phase-0 Results**: ✅ COMPLETE
- Long VX2: PASS (MaxDD +0.91%, Worst-month +0.49%)
- Long VX2 - VX1 spread: PASS (MaxDD +0.76%, Worst-month +0.41%)
- Long UB: PASS (parked for post-v1 reconsideration)

**Phase-1 Results** (2025-12-17): ✅ COMPLETE
- **Long VX3**: ✅ **PROMOTED** to Phase-2 (MaxDD +0.41%, Worst-month +0.23%, CAGR +0.17% vs Core)
- **Long VX2**: Retained as benchmark ceiling reference (not promoted)
- **Long VX2 - VX1 spread**: ❌ **PARKED** (Phase-1 FAIL — insufficient tail preservation)
- **Long VX3 - VX2 spread**: ❌ **PARKED** (Phase-1 FAIL — insufficient tail preservation)

**Phase-2 Results** (2025-12-17): ❌ FAIL
- Overall tail metrics improved vs Core v9 (MaxDD, worst-month, worst-quarter, 2022 drawdown)
- But 2020 Q1 fast-crash window showed portfolio-level drawdown deterioration
- Instrument-level VX behavior is correct (VX1 > VX2 > VX3 in 2020 Q1), but portfolio-level integration fails

**Final Decision**: Crisis Meta-Sleeve v1 — NO PROMOTION
- Long VX3 fails Phase-2 due to deterioration in 2020 Q1 fast-crash behavior
- Always-on convexity not adopted in v1
- Crisis protection deferred to allocator logic (v2+)

**Disposition:**
- **VX2**: Retained as benchmark only (not promoted)
- **VX3**: Parked as "tail smoother candidate" for allocator-era research
- **UB**: Remains parked (conditional hedge, post-v1 reconsideration)
- **VX Spreads**: Parked (Phase-1 FAIL)

**Post-v1 / Allocator Track:**
Crisis protection to be implemented as allocator-driven convexity allocation (e.g., conditional VX exposure, potential VX2/VX3 blending) subject to allocator research lifecycle.

**Optional Parked Ideas** (non-committal, allocator-era research):
- Conditional convexity activation
- VX2/VX3 blend rules
- Shock/vol regime detection
- Options-based convexity (explicitly deferred, high complexity)

**Future Candidate Atomic Sleeves (Post-Phase-0):**
- VIX convexity sleeve (long VIX calls or synthetic convexity via VX futures curvature)
- VX curve inversion sleeve (crisis filter)
- Equity skew sleeve
- Vol-of-vol sleeve (VVIX or alternative derived data)
- Synthetic straddle/strangle-like proxies using futures spreads

**Architectural Guardrail:** Crisis Meta-Sleeve is a distinct economic engine and will not be blended or hybridized with Carry or Curve RV during Phase-0/1.

### 4.5 Production Release (Target: After All Pre-Phase-B Meta-Sleeves)

Once Trend, CSMOM, VRP, Carry, Curve RV (Phase-0), and Crisis (minimal) are complete:

- System is eligible for first production deployment.

**Production includes:**

- Canonicalized data pipelines
- Allocator Layer v1
  - ERC or Ridge-RP
- Portfolio-level volatility targeting
- Live signal generation
- Logging + run consistency checks
- Monitoring dashboards
  - rolling Sharpe
  - rolling correlations
  - sleeve contributions
  - drawdown attribution

**Sharpe target for first production release:**
0.8–1.1 unlevered (expected with complete pre-Phase-B meta-sleeve set)

### 4.6 Post-v1 Roadmap: Allocator & Production

**Explicit Sequencing:**

Futures-Six intentionally launches with a minimum viable but architecturally complete system. Additional sleeves and refinements are accretive and governed post-launch.

**Ordered Roadmap:**

1. **Allocator v1 (rule-based, deterministic)**
   - Allocator Phase-A: Architecture & invariants (no tuning)
   - Allocator Phase-B: Deterministic rule-based allocator
   - Allocator Phase-C: End-to-end validation

2. **Expanded historical validation (structural window)**
   - Validate system on expanded historical window (e.g., 2010–present)
   - Structural validation (not parameter optimization)
   - Failure mode discovery and architecture refinement

3. **End-to-end production readiness checks**
   - Historical stress windows (2020 Q1, 2022, etc.)
   - Sleeve-level loss attribution
   - Allocator behavior under stress
   - Correlation spikes and path-dependent drawdowns
   - Tooling hardening: committee-pack generator + batch generator + pinned baselines for ablation matrix

4. **Production freeze (Core v9)**
   - Freeze production engines
   - Freeze allocator logic
   - Version control and deployment

5. **Capital deployment**
   - Deploy production system with real capital
   - Establish monitoring and alerting

6. **Parallel research track (Core v10 candidates)**
   - New sleeves developed in research track
   - New allocator logic developed in research track
   - Paper integration with production copy
   - Formal promotion process for additions

---

## 5. Medium-Term Roadmap (Post-Production Expansion Cycle)

**Goal: Increase Sharpe to institutional target range (1.5–2.0).**

After production deployment, the next development wave focuses on Phase-B optimization and allocator work:

### 5.1 Allocator Development Sequencing

**Post-Production Allocator Work:**

Allocator development follows the structured lifecycle defined in `PROCEDURES.md` § "Allocator Development Lifecycle (v2 Track)":

1. **Allocator Phase-A**: Architecture & invariants (no tuning)
   - Define allocator architecture
   - Establish invariants
   - Design conditional logic structure

2. **Allocator Phase-B**: Deterministic rule-based allocator
   - Implement rule-based allocator
   - Validate deterministic behavior
   - No optimization at this stage

3. **Allocator Phase-C**: End-to-end validation
   - Validate across expanded historical window
   - Stress test failure modes
   - Portfolio-level validation

4. **Allocator Phase-D**: Production deployment
   - Freeze allocator logic
   - Version control (e.g., Allocator v1)
   - Deploy to production

5. **Allocator Phase-E**: Post-production enhancements
   - Iterative improvements in research track
   - Formal promotion process
   - Versioned releases (e.g., Allocator v2)

#### 5.1.1 Allocator v1 Status (December 2024)

**Status:** Phase-D COMPLETE (production-ready)

See [docs/SOTs/SYSTEM_CONSTRUCTION.md](docs/SOTs/SYSTEM_CONSTRUCTION.md) for allocator architecture, layers, and implementation. See [docs/SOTs/PROCEDURES.md](docs/SOTs/PROCEDURES.md) for allocator promotion and two-pass audit workflow.

### 5.2 Phase-B Optimization (Post-Production)

**Optimization Scope:**
- Meta-sleeve weight optimization
- Correlation-aware allocation
- Turnover and cost management

**Note**: Optimization occurs only after allocator architecture is validated and production system is frozen.

### 5.2 Additional Meta-Sleeves (Optional, Long-Term)

#### 5.2.1 Value Meta-Sleeve

Economic risk premia via:

- mean-reversion signals
- value spreads in commodities
- deep carry interactions
- bond valuation signals

#### 5.2.2 Macro Regime Sleeve

- Yield curve steepening/flattening
- Inflation momentum
- ISM/PPI/credit spreads filters

Used only as conditional overlays, not direct forecasts

---

## 6. Long-Term Roadmap (2–3 years)

**Goal: 7–10 atomic sleeves per meta-sleeve**

Mirroring AQR / Man-AHL architecture.

**Target distribution:**

| Meta-Sleeve | Target # Atomic Sleeves |
|-------------|-------------------------|
| Trend | 7–10 |
| CSMOM | 3–5 |
| VRP | 5–7 |
| Carry | 5–7 |
| Curve RV | 3–5 |
| Crisis | 5–8 |
| Macro (optional) | 3–5 |

---

## 7. Sharpe Targets (Strategic)

**Target 1 — Pre-Production:**
- Sharpe 0.8–1.1

**Target 2 — Post Carry Sleeve:**
- Sharpe 1.1–1.3

**Target 3 — Post Curve RV + Crisis Sleeves:**
- Sharpe 1.4–1.7

**Target 4 — Full 6 Meta-Sleeve System (Pre-Phase-B):**
- Sharpe 1.6–2.0

**Based on discussions and empirical evidence:**

- Trend contributes ~0.5–0.7
- VRP contributes ~0.2–0.3
- Carry contributes ~0.1–0.25
- Curve RV contributes ~0.05–0.15
- Crisis/convexity contributes 0.15–0.30
- CSMOM adds ~0.05–0.10

**Total ≈ 1.2–1.6 unlevered before Phase-B optimization.**

---

## 8. Dependencies & Sequencing Rules

**Pre-Phase-B Meta-Sleeve Sequencing:**
- Trend and CSMOM complete → VRP can be added
- VRP must be complete → Carry Meta-Sleeve can start
- Carry Meta-Sleeve in progress → Curve RV Meta-Sleeve Phase-2 complete ✅ → Crisis Meta-Sleeve (minimal) can begin
- All pre-Phase-B meta-sleeves complete → Production release eligible

**Post-Production:**
- Production release → Phase-B optimization and allocator work begins
- Sharpe target progress → Additional meta-sleeves added (optional)

**Architectural Guardrail:** Carry, Curve RV, and Crisis Meta-Sleeves are distinct economic engines and will not be blended or hybridized during Phase-0/1. Allocator logic, regime conditioning, and optimization remain Phase-B objectives, after all core Meta-Sleeves exist.

---

## 9. Risks & Monitoring Focus

- VRP Sharpe underperformance
- Correlation spikes between sleeves
- Crisis regime degradation
- Overfitting in Phase-1 sleeves
- Warmup window fragility
- Canonical data source drift
- Target volatility mismatch across sleeves

---

## 10. Roadmap Summary (Executive)

**Step 1: Finish VRP Meta-Sleeve**
- VRP-Core ✅
- VRP-Convergence ✅
- VRP-RollYield ❌ (PARKED)
- VRP-TermStructure ❌ (PARKED)
- (Optional) VRP-CalendarDecay
- Meta-sleeve assembly

**Step 2: Build Carry Meta-Sleeve**
- 3–4 atomic sleeves
- Promote and integrate
- Improve Sharpe + diversification

**Step 3: Production Deployment**
- Allocator
- Vol targeting
- Monitoring
- Deterministic pipeline

**Step 4: Post-Production Wave**
- Crisis Meta-Sleeve
- Value Meta-Sleeve
- Macro overlays

**Step 5: Long-Term Expansion**
- Move toward 7–10 atomic sleeves per meta-sleeve
- Institutional Sharpe target (1.5–2.0)

