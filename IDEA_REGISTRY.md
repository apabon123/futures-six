# IDEA_REGISTRY.md

**Futures-Six Trading Ideas Archive & Classification**
**Last Updated:** 2026-02-03
**Purpose:** Preserve and cluster all trading/research ideas with forward-looking research questions

---

## Registry Overview

**Total Ideas Cataloged:** 38
**Promoted to Production:** 11
**Parked/Failed:** 14
**Planned/Future:** 13

**Source Material:**
- `docs/SOTs/ROADMAP.md` (strategic development plan)
- `docs/SOTs/STRATEGY.md` (implementation details)
- `docs/META_SLEEVES/` (meta-sleeve research)
- Root-level carry memos and phase documentation
- `reports/phase4_research/` (hypothesis testing)

---

## Active/Promoted Ideas (Production or Research-Ready)

### IDEA-001: Time-Series Momentum (TSMOM) — Long Horizon

**Title:** Long-Term Time-Series Momentum (252-Day)

**Source Files:**
- `docs/META_SLEEVES/TREND_IMPLEMENTATION.md`
- `docs/META_SLEEVES/TREND_RESEARCH.md`
- `docs/SOTs/STRATEGY.md` (lines 409-423)
- `src/agents/strat_tsmom_multihorizon.py`

**Asset Class:** Multi-asset (Equity indices, Rates, FX, Commodities)

**Strategy Family:** Trend / Time-Series Momentum

**Description:**
Captures persistent directional price trends over a 12-month horizon (252 trading days). Signal combines 252d cumulative returns, 252d Donchian breakouts, and slow trend slope indicators. Equal-weighted ensemble (1/3, 1/3, 1/3) of three features. Long positive momentum assets, short negative momentum assets.

**Signals/Drivers:**
- 252-day log returns (primary signal)
- 252-day Donchian channel breakouts
- Slow trend slope (long-period EMA crossovers)

**Historical Implementation:**
- Phase-0: Sign-only signals, equal-weight across assets
- Phase-1: Vol-normalized, cross-sectional z-scoring
- Phase-2: Integrated into Core v3+ baselines
- Currently: 48.5% weight within Trend Meta-Sleeve

**Candidate Modern Implementation:**
- Adaptive lookback periods (regime-dependent)
- Vol-of-vol aware scaling (reduce exposure in high-vol regimes)
- Asset-class specific parameterization

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (52.4% × 48.5% = 25.4% effective weight)

**Classification Justification:**
Foundational component of Trend Meta-Sleeve. Passed Phase-0/1/2 validation. Strong Sharpe contribution (approximately 0.3-0.4 standalone). Documented in STRATEGY.md as canonical long-horizon implementation.

**Extraction Confidence:** High

**What would invalidate this idea?**
- Persistent negative Sharpe over 5+ years (structural regime change)
- Correlation with short-term trend exceeds 0.85 (redundancy)
- Drawdowns exceed -50% in single year (unacceptable tail risk)
- Crisis behavior worse than -30% in Q1 2020-style events
- Signal degeneracy (>80% of days in same direction)

**Research questions to verify in 2026:**
- Does skip window (e.g., 21d gap) improve Sharpe in high-vol regimes?
- How does 252d momentum correlate with macro regime features (VIX, yield curve)?
- Is there asset-class heterogeneity in optimal lookback (commodities vs rates)?
- Can volatility-adjusted position sizing reduce drawdowns without sacrificing CAGR?
- What is the optimal rebalance frequency (daily vs weekly vs monthly)?
- Does ensemble weighting (1/3, 1/3, 1/3) dominate other schemes?

**Needs web research?** No

**Web research priority:** N/A

---

### IDEA-002: Time-Series Momentum (TSMOM) — Medium Horizon

**Title:** Medium-Term Time-Series Momentum (84/126-Day)

**Source Files:**
- `docs/SOTs/STRATEGY.md` (lines 410-411)
- `docs/META_SLEEVES/TREND_IMPLEMENTATION.md`
- Legacy: `docs/legacy/TSMOM_IMPLEMENTATION.md`

**Asset Class:** Multi-asset (Equity indices, Rates, FX, Commodities)

**Strategy Family:** Trend / Time-Series Momentum

**Description:**
Captures intermediate directional price trends over 3-6 month horizons. Legacy production uses 84d and 126d lookbacks. Canonical research version uses 84d single horizon with equal-weight feature ensemble. Signal includes 84d returns, 84d breakouts, and medium trend slope (EMA21-84).

**Signals/Drivers:**
- 84-day log returns
- 126-day breakouts (legacy)
- Medium trend slope (EMA_20 - EMA_84)
- Persistence features

**Historical Implementation:**
- Phase-1: Equal-weight canonical (1/3, 1/3, 1/3) with 84d, skip 10d/21d, 21d vol scaling
- Production: Legacy dual-horizon (84d/126d) with multiple features
- Currently: 29.1% weight within Trend Meta-Sleeve

**Candidate Modern Implementation:**
- Consolidate to single 84d canonical (simplify)
- Add skip windows (10d or 21d) to reduce reversal contamination
- Cross-sectional z-scoring for asset-relative strength

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (52.4% × 29.1% = 15.2% effective weight)

**Classification Justification:**
Second-largest component of Trend Meta-Sleeve. Provides diversification vs long-term momentum (correlation <0.8). Legacy version in production; canonical version validated in Phase-1 research.

**Extraction Confidence:** High

**What would invalidate this idea?**
- Correlation with 252d trend exceeds 0.90 (redundancy)
- Standalone Sharpe drops below 0.0 for 3+ years
- Crisis losses exceed -40% in Q1 2020-style events
- Signal turnover exceeds 200% annually (transaction cost erosion)

**Research questions to verify in 2026:**
- Is 84d optimal, or should it be 63d or 126d?
- Does skip window improve Sharpe (current research: 10d/21d skip)?
- How does medium momentum interact with VRP sleeve (correlation during vol spikes)?
- Can EWMA smoothing reduce whipsaws without lagging too much?
- Is persistence feature additive, or redundant with returns?

**Needs web research?** No

**Web research priority:** N/A

---

### IDEA-003: Time-Series Momentum (TSMOM) — Short Horizon

**Title:** Short-Term Time-Series Momentum (21-Day)

**Source Files:**
- `docs/SOTs/STRATEGY.md` (lines 412-413)
- `docs/META_SLEEVES/TREND_IMPLEMENTATION.md`

**Asset Class:** Multi-asset (Equity indices, Rates, FX, Commodities)

**Strategy Family:** Trend / Time-Series Momentum

**Description:**
Captures short-term directional price trends over 1-month horizon (21 trading days). Signal includes 21d returns, 21d breakouts, and fast trend slope (EMA_10 - EMA_40). Optional reversal filter to avoid mean-reversion regimes. High turnover but responsive to rapid regime shifts.

**Signals/Drivers:**
- 21-day log returns (primary signal)
- 21-day Donchian breakouts
- Fast trend slope (short-period EMA crossovers)
- Optional reversal filter

**Historical Implementation:**
- Legacy production: 0.5/0.3/0.2 feature weights
- Canonical research: 1/3, 1/3, 1/3 equal weights
- Currently: 19.4% weight within Trend Meta-Sleeve

**Candidate Modern Implementation:**
- Add skip window (5d) to reduce overnight reversal noise
- Adaptive position sizing (lower weight in high-vol regimes)
- Optional reversal filter activation in crisis periods

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (52.4% × 19.4% = 10.2% effective weight)

**Classification Justification:**
Smallest component of Trend Meta-Sleeve but still material. Provides fast response to regime shifts. Higher turnover than long/medium horizons. Validated in Phase-0/1/2.

**Extraction Confidence:** High

**What would invalidate this idea?**
- Negative Sharpe over 3+ years (mean-reversion dominates)
- Transaction costs exceed alpha (turnover >300% annually)
- Correlation with 84d trend exceeds 0.95 (redundancy)
- Reversal contamination: >50% of signals reverse within 5 days

**Research questions to verify in 2026:**
- Does 21d lookback dominate 15d or 30d alternatives?
- Is skip window (5d) necessary for short-term momentum?
- How does short-term momentum behave in flash crash events (2020-03, 2022-02)?
- Can reversal filter improve Sharpe without reducing exposure too much?
- What is the correlation with VIX spikes (contemporaneous vs lagged)?

**Needs web research?** Yes

**Web research priority:** Medium (literature on short-term momentum and reversal effects)

---

### IDEA-004: Residual Trend (Long-Short Dual-Horizon)

**Title:** Residual Trend (252d Long - 21d Short)

**Source Files:**
- `docs/SOTs/STRATEGY.md` (lines 412, 421)
- `docs/META_SLEEVES/TREND_IMPLEMENTATION.md`

**Asset Class:** Multi-asset (Equity indices, Rates, FX, Commodities)

**Strategy Family:** Trend / Mean-Reversion Hybrid

**Description:**
Isolates the "structural" trend by going long 252d momentum while simultaneously shorting 21d momentum. Captures mean-reversion within longer-term trends. Cross-sectionally z-scored for asset-relative strength. Validated in Phase-1 as diversifier to pure momentum sleeves.

**Signals/Drivers:**
- Long-horizon trend signal (252d returns)
- Minus short-horizon trend signal (21d returns)
- Cross-sectional z-scoring (relative strength across assets)

**Historical Implementation:**
- Phase-1: Validated as standalone atomic sleeve
- Not yet integrated into Core v9 baseline (research-only)
- Intended as optional 5th atomic sleeve within Trend Meta

**Candidate Modern Implementation:**
- Add to Core v10 baseline at 5-10% weight within Trend Meta
- Test optimal horizon pairs (e.g., 126d-21d, 252d-63d)
- Conditional activation (only in trending regimes, not sideways markets)

**Compatibility:**
✅ **Futures-Six candidate** — Validated in Phase-1, not yet promoted to Core

**Classification Justification:**
Passed Phase-1 validation but not yet integrated into Core baseline. ROADMAP.md mentions it as validated atomic sleeve. Provides diversification vs pure momentum (captures mean-reversion within trends).

**Extraction Confidence:** High

**What would invalidate this idea?**
- Correlation with long-term trend exceeds 0.90 (redundancy)
- Standalone Sharpe consistently negative over 3+ years
- Crisis behavior worse than -20% (2020 Q1 benchmark)
- Signal degeneracy: >70% of days in same direction

**Research questions to verify in 2026:**
- Is 252d-21d optimal, or should it be 126d-21d or 252d-63d?
- Does this sleeve work better in sideways markets or trending markets?
- What is the correlation with CSMOM (both are cross-sectional)?
- Can position sizing improve Sharpe (e.g., scale down when both signals agree)?
- How does this perform during 2020 and 2022 drawdowns?

**Needs web research?** Yes

**Web research priority:** Medium (literature on dual-momentum and trend-reversion hybrids)

---

### IDEA-005: Breakout Mid (50-100d Donchian)

**Title:** Medium-Horizon Breakout Strategy (50-100d)

**Source Files:**
- `docs/SOTs/STRATEGY.md` (lines 413-414, 421-422)
- `docs/META_SLEEVES/TREND_IMPLEMENTATION.md`

**Asset Class:** Multi-asset (Equity indices, Rates, FX, Commodities)

**Strategy Family:** Trend / Breakout

**Description:**
Trades Donchian channel breakouts over 50-100 day horizons. Long when price breaks above 50d/100d high, short when breaks below 50d/100d low. Feature ensemble uses 70/30 weights. Passed Phase-0/1B/2/3 validation. Integrated into Core v3+ baselines at 3% weight within Trend Meta.

**Signals/Drivers:**
- 50-day Donchian high/low
- 100-day Donchian high/low
- Weighted ensemble (70/30)

**Historical Implementation:**
- Phase-0/1B/2/3: Validated with 70/30 feature weights
- Currently: 3% weight within Trend Meta-Sleeve
- Production: Integrated into `core_v3_no_macro` profile

**Candidate Modern Implementation:**
- Adaptive channel width (based on realized volatility)
- Regime-dependent activation (disable in sideways markets)
- Asset-class specific parameterization (commodities may need wider channels)

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (52.4% × 3% = 1.6% effective weight)

**Classification Justification:**
Smallest component of Trend Meta-Sleeve. Passed full Phase-0/1B/2/3 validation. STRATEGY.md explicitly states "passed Phase-0/1B/2/3 validation (70/30 config, 3% weight)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Correlation with 84d momentum exceeds 0.95 (redundancy)
- Standalone Sharpe drops below -0.2 for 2+ years
- Transaction costs exceed alpha (breakouts trigger frequent trades)
- Crisis behavior: drawdowns exceed -30% in single year

**Research questions to verify in 2026:**
- Is 50-100d optimal, or should it be 40-80d or 60-120d?
- Can adaptive channel width improve Sharpe (e.g., ATR-based)?
- How does breakout perform in sideways vs trending regimes?
- What is the optimal feature weight (70/30 vs 50/50 vs 80/20)?
- How does this correlate with VIX spikes?

**Needs web research?** Yes

**Web research priority:** Low (Donchian breakouts are well-documented)

---

### IDEA-006: Cross-Sectional Momentum (CSMOM) — Multi-Horizon

**Title:** Cross-Sectional Momentum (63/126/252d Lookbacks)

**Source Files:**
- `docs/SOTs/STRATEGY.md` (lines 427-473)
- `docs/components/strategies/CROSS_SECTIONAL_MOMENTUM.md`
- `reports/phase4_research/csmom_hypotheses/HYPOTHESIS_001_CSMOM_SKIP_V1.md`
- `reports/phase4_research/csmom_phase2/CSMOM_skip_v1_phase2_results_20260120_115139.md`

**Asset Class:** Multi-asset (Equity indices, Rates, FX, Commodities)

**Strategy Family:** Cross-Sectional Momentum

**Description:**
Ranks assets by relative momentum (winners vs losers) and goes long top performers, short bottom performers. Market-neutral positioning (zero-sum across universe). Multi-horizon ensemble (63d/126d/252d) with weights [0.4, 0.35, 0.25]. Vol-tempered and cross-sectionally neutralized. Phase 4 enhancement adds skip windows (5/10/21d) to reduce reversal contamination.

**Signals/Drivers:**
- Cross-sectional ranking by cumulative returns (63d, 126d, 252d)
- Volatility tempering (divide by 63d realized vol)
- Cross-sectional neutralization (zero-sum)
- Skip windows: 5d, 10d, 21d (CSMOM_skip_v1, promoted Jan 2026)

**Historical Implementation:**
- Phase-0: Sign-only cross-sectional ranks
- Phase-1: Multi-horizon ensemble with vol-aware ranking
- Phase-2: Integrated into Core v4+ baselines (25% → 21.85% in Core v9)
- Phase 4: Skip windows added (CSMOM_skip_v1), promoted to Core v10 candidate

**Candidate Modern Implementation:**
- Skip windows (5/10/21d) now canonical (CSMOM_skip_v1)
- Sector-neutral variants (within commodities, rates, FX separately)
- Adaptive horizon weights (shift toward shorter horizons in high-vol regimes)

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (21.85% weight)

**Classification Justification:**
Second-largest meta-sleeve in Core v9. Passed Phase-0/1/2 validation. Phase 4 enhancement (skip windows) promoted Jan 2026 based on +0.06 Post-Construction Sharpe improvement. ROADMAP.md: "CSMOM_skip_v1 — PROMOTED (2026-01-20)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Universe contraction below 10 assets (ranking becomes degenerate)
- Correlation with TSMOM exceeds 0.80 in all regimes (redundancy)
- Crisis losses exceed -20% in single quarter (2020 Q1 benchmark)
- Standalone Sharpe drops below -0.5 for 2+ years (structural failure)
- Skip windows increase latency without improving Sharpe

**Research questions to verify in 2026:**
- Are skip windows (5/10/21d) optimal, or should they be (3/7/14d)?
- How does CSMOM perform with expanded universe (30+ assets vs current 13)?
- Can sector-neutral variants reduce single-asset concentration risk?
- What is the optimal rebalance frequency (daily vs weekly)?
- How does CSMOM interact with macro regime filter (FRED indicators)?
- Does vol-tempering help or hurt in crisis regimes (2020 Q1, 2022)?

**Needs web research?** Yes

**Web research priority:** High (academic literature on skip windows, momentum crashes, and universe scalability)

---

### IDEA-007: VRP-Core (VIX - RV21)

**Title:** Volatility Risk Premium — Core (VIX vs 21d Realized Vol)

**Source Files:**
- `docs/SOTs/STRATEGY.md` (lines 39-40, 279-280, 332-341)
- `docs/SOTs/ROADMAP.md` (lines 461-467, 481-483, 727-728)
- `docs/SOTs/DIAGNOSTICS.md` (referenced extensively)

**Asset Class:** Volatility (VIX futures: VX1)

**Strategy Family:** Volatility Risk Premium / Option Selling Proxy

**Description:**
Captures the persistent premium between implied volatility (VIX) and realized equity volatility (21d ES realized vol). Short VX1 when VIX > RV21 (z-scored over 252d window). Directional VX1 trading, not spread. Systematic insurance selling strategy. First canonical VRP sleeve.

**Signals/Drivers:**
- VRP spread: VIX - RV21 (ES 21-day realized volatility)
- Z-scored over 252-day rolling window
- Signal: Short VX1 when z-score positive

**Historical Implementation:**
- Phase-0: Sign-only (VIX > RV21 → short VX1)
- Phase-1: Z-scored spread with 252d normalization window
- Phase-2: Integrated into Core v5 baseline (10% → 7.5% → 6.555% in Core v9)
- Phase 4: Audited, no changes (VRP-Convergence removed, not VRP-Core)

**Candidate Modern Implementation:**
- Adaptive position sizing (scale down when VVIX elevated)
- Crisis filter (reduce exposure when VIX > 30)
- Vol-of-vol awareness (scale less when VIX jumps are frequent)

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (6.555% weight)

**Classification Justification:**
First VRP sleeve promoted to production. ROADMAP.md: "VRP-Core ✅ Complete (Phase-2 integrated) — First canonical VRP sleeve." STRATEGY.md: "VRP-Core passed Phase-0 (toy econ test), Phase-1 (engineered sleeve), and Phase-2 (portfolio integration)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- VIX term structure inverts permanently (backwardation becomes structural)
- VX futures liquidity deteriorates (bid-ask spreads widen significantly)
- Realized vol exceeds VIX for 50%+ of days (volatility risk premium disappears)
- Crisis losses exceed -40% in single year (convexity blow-up)
- Correlation with short equity trend exceeds 0.90 (redundant with trend sleeve)

**Research questions to verify in 2026:**
- Has the VIX-RV21 spread mean reverted toward zero in recent years (2024-2026)?
- Can VVIX (vol-of-vol) improve signal quality as an overlay?
- What is the optimal z-score window (252d vs 126d vs 504d)?
- How does VRP-Core perform during vol spikes (>40 VIX)?
- Can adaptive position sizing reduce drawdowns without sacrificing CAGR?
- What is the transaction cost impact of daily rebalancing?

**Needs web research?** Yes

**Web research priority:** High (VIX term structure evolution, post-COVID vol regime changes)

---

### IDEA-008: VRP-Alt (VIX - RV5)

**Title:** Volatility Risk Premium — Alt (VIX vs 5d Realized Vol)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 487-511, 727-729)
- `docs/SOTs/STRATEGY.md` (lines 280-281, 344-351)

**Asset Class:** Volatility (VIX futures: VX1)

**Strategy Family:** Volatility Risk Premium / Option Selling Proxy

**Description:**
Alternative VRP signal using shorter-horizon realized volatility (5d ES realized vol instead of 21d). Front-month futures tend to be overpriced relative to very short-term realized vol, capturing spike decay. More sensitive to vol spikes than VRP-Core. Z-scored over 252d window. Second canonical VRP sleeve (promoted Dec 2025).

**Signals/Drivers:**
- VRP spread: VIX - RV5 (ES 5-day realized volatility)
- Z-scored over 252-day rolling window
- Signal: Short VX1 when z-score positive

**Historical Implementation:**
- Phase-0: Borderline (not documented, implied from roadmap)
- Phase-1: Strong performance (z-scored spread)
- Phase-2: Inconclusive but passed integration tests
- Scaling verification: Promoted at 15% weight (Dec 2025)
- Currently: 13.11% weight in Core v9

**Candidate Modern Implementation:**
- Combine with VRP-Core in meta-sleeve (already done)
- Adaptive weighting (shift toward RV5 during spike decay periods)
- Crisis filter (reduce exposure when VIX spikes >20% in single day)

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (13.11% weight)

**Classification Justification:**
Second canonical VRP sleeve. ROADMAP.md: "VRP-Alt ✅ PROMOTED (Dec 2025) — Core v7 at 15% weight." STRATEGY.md: "VRP-Alt passed Phase-0 (borderline), Phase-1 (strong), Phase-2 (inconclusive), and scaling verification (promoted at 15% weight)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Correlation with VRP-Core exceeds 0.95 (redundancy)
- Negative Sharpe over 3+ years (short-term RV no longer predictive)
- Crisis losses exceed -50% in single year (convexity blow-up worse than Core)
- Signal degeneracy: >80% of days in same direction as VRP-Core

**Research questions to verify in 2026:**
- Is RV5 optimal, or should it be RV3 or RV10?
- How does VRP-Alt perform during flash crashes (2020-03-12, for example)?
- What is the correlation with VRP-Core across different vol regimes?
- Can adaptive weighting (Core vs Alt) improve meta-sleeve Sharpe?
- Does VRP-Alt capture "spike decay" alpha better than Core?

**Needs web research?** Yes

**Web research priority:** Medium (literature on short-term realized vol and VIX term structure)

---

### IDEA-009: SR3 Calendar Carry (SOFR R2-R1 Spread)

**Title:** SOFR Futures Calendar Carry (R2-R1 Spread)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 776-779)
- `docs/SOTs/STRATEGY.md` (lines 482-491)
- `docs/SOTs/DIAGNOSTICS.md` (referenced)
- `docs/components/strategies/SR3_CARRY_CURVE.md`
- `src/strategies/carry/sr3_calendar_carry.py`

**Asset Class:** Rates (SOFR futures)

**Strategy Family:** Carry / Roll Yield

**Description:**
Captures roll yield from SOFR futures term structure. Short calendar spread (R2-R1) when contango exists (positive roll yield). Z-scored signal over 252d window. Trade at close, P&L accrues close-to-close. Risk stabilizer and diversifier vs Trend/VRP. Promoted Dec 2025 at 5% research weight (Core v8, then 4.6% in Core v9).

**Signals/Drivers:**
- SOFR futures R2-R1 spread (term structure slope)
- Z-scored over 252d rolling window
- Signal: Short spread when contango (positive roll yield)

**Historical Implementation:**
- Phase-0: PASS (Sharpe 0.6384, R2-R1 canonical pair)
- Phase-1: PASS (z-scoring, vol targeting, execution rules frozen)
- Phase-2: PASS (MaxDD improvement +0.80%, correlation 0.04, Sharpe preserved)
- Currently: 4.6% weight in Core v9 (scaled from 5% in Core v8)

**Candidate Modern Implementation:**
- Already implemented as canonical (no major changes proposed)
- Optional: Add multiple contract pairs (R3-R2, R4-R3) for diversification
- Optional: Regime-dependent activation (disable in rate-shock regimes)

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (4.6% weight)

**Classification Justification:**
Promoted carry sleeve. ROADMAP.md: "SR3 Calendar Carry — ✅ PROMOTED (Dec 2025)." STRATEGY.md: "SR3 Calendar Carry passed Phase-0, Phase-1, and Phase-2 (promoted at 5% weight)." DIAGNOSTICS.md: "Full development history and promotion decision."

**Extraction Confidence:** High

**What would invalidate this idea?**
- SOFR term structure inverts for 50%+ of days (backwardation becomes structural)
- Correlation with short-term rates exceeds 0.90 (redundant with rates trend)
- Crisis losses exceed -20% in single year (carry unwind too severe)
- Transaction costs exceed alpha (spread trades have wider bid-ask)

**Research questions to verify in 2026:**
- Has SOFR term structure changed post-2023 (Fed pivot impact)?
- Can multiple contract pairs (R3-R2, R4-R3) improve diversification?
- What is the optimal z-score window (252d vs 126d vs 504d)?
- How does SR3 carry perform during Fed policy pivots (rate cuts/hikes)?
- Can adaptive position sizing reduce drawdowns during rate shocks?

**Needs web research?** Yes

**Web research priority:** Medium (SOFR term structure dynamics, Fed policy impact on carry)

---

### IDEA-010: VX Calendar Carry (VX2-VX1 Short Spread)

**Title:** VIX Futures Calendar Carry (VX2-VX1 Short Spread)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 782-793)
- `docs/SOTs/STRATEGY.md` (lines 492-500)
- `docs/SOTs/DIAGNOSTICS.md` (referenced)
- `src/strategies/carry/vx_calendar_carry.py`

**Asset Class:** Volatility (VIX futures)

**Strategy Family:** Carry / Roll Yield / Volatility

**Description:**
Captures volatility term carry by shorting VX calendar spreads when contango exists. Short VX2-VX1 spread when term structure is upward-sloping (positive roll yield). Z-scored signal over 252d window. Canonical atomic sleeve (VX2-VX1_short); secondary sleeve (VX3-VX2_short) validated but non-default. Portfolio glue with low correlation to Trend/VRP.

**Signals/Drivers:**
- VX2-VX1 spread (term structure slope)
- Z-scored over 252d rolling window
- Signal: Short spread when contango (VX2 > VX1)

**Historical Implementation:**
- Phase-0: PASS (both short-spread variants show strong standalone carry expectancy)
- Phase-1: PASS (z-scoring, vol targeting, execution rules frozen; two atomic sleeves)
- Phase-2: PASS (Sharpe +0.0377, MaxDD +1.02%, Vol -0.71%, correlation -0.0508)
- Promoted: VX2-VX1_short canonical (Dec 2025), integrated into Core v8 at 5% (4.6% in Core v9)

**Candidate Modern Implementation:**
- Already implemented as canonical (VX2-VX1_short)
- Optional: Switch to VX3-VX2_short if liquidity improves (validated backup)
- Optional: Dynamic spread selection (switch based on liquidity or vol regime)

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (4.6% weight)

**Classification Justification:**
Promoted carry sleeve. ROADMAP.md: "VX Calendar Carry — ✅ COMPLETE / PROMOTED (Dec 2025)." STRATEGY.md: "VX Carry uses canonical atomic sleeve: VX2–VX1_short." DIAGNOSTICS.md: "Full development history and promotion decision."

**Extraction Confidence:** High

**What would invalidate this idea?**
- VIX term structure inverts for 50%+ of days (backwardation becomes structural)
- VX futures liquidity deteriorates significantly (bid-ask spreads widen)
- Correlation with VRP-Core exceeds 0.90 (redundant with VRP directional)
- Crisis losses exceed -30% in single year (carry unwind too severe)

**Research questions to verify in 2026:**
- Has VIX term structure changed post-2023 (persistent backwardation)?
- Can VX3-VX2 spread improve diversification vs VX2-VX1?
- What is the optimal z-score window (252d vs 126d)?
- How does VX carry perform during vol spikes (>40 VIX)?
- Can dynamic spread selection (VX2-VX1 vs VX3-VX2) improve Sharpe?

**Needs web research?** Yes

**Web research priority:** Medium (VIX term structure evolution, post-COVID vol regime)

---

### IDEA-011: Curve RV — Rank Fly Momentum (SR3 2-6-10 Fly)

**Title:** SOFR Curve Shape Momentum — Rank Fly (2s6s10s)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 815-914)
- `docs/SOTs/STRATEGY.md` (lines 187-191, 318, 363-369)
- `docs/components/strategies/SR3_CARRY_CURVE.md`
- `src/strategies/rates_curve_rv/sr3_curve_rv_momentum.py`

**Asset Class:** Rates (SOFR futures)

**Strategy Family:** Curve RV / Momentum-Driven Regime

**Description:**
Captures persistent regimes in yield curve shape through momentum-driven signals (not mean-reversion). Rank fly construction: 2*r6 - r2 - r10 (where r = 100 - price). Long fly when positive momentum, short when negative. Market-neutral spread construction. Policy-driven short-rate regime detection. Promoted Dec 2025 at 5% weight (Core v9).

**Signals/Drivers:**
- Rank fly: 2*r6 - r2 - r10 (spread/fly construction in rate space)
- Momentum on fly shape: sign(fly_today - fly_lagged)
- Execution lag: signal.shift(1) (frozen)

**Historical Implementation:**
- Phase-0: PASS (Sharpe 0.81, momentum variant; mean-reversion variants failed -0.42 to -0.81)
- Phase-1: PASS (Sharpe 1.19, 63d momentum lookback)
- Phase-2: PASS (integrated into Core v9 at 5% weight)
- Currently: 5% weight in Core v9

**Candidate Modern Implementation:**
- Already implemented as canonical (no major changes proposed)
- Optional: Multiple fly structures (2s5s10s, 5s10s30s) for diversification
- Optional: Adaptive lookback (shorter in high-vol regimes)

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (5% weight)

**Classification Justification:**
Promoted curve sleeve. ROADMAP.md: "Rank Fly (2-6-10) — ✅ PROMOTED" with "Core v9 weight: 5%." STRATEGY.md: "Curve RV Meta-Sleeve (8%: Rank Fly 5% + Pack Slope 3%) [...] both atomics integrated into Core v9."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Correlation with Pack Slope exceeds 0.95 (redundancy within Curve RV meta-sleeve)
- Negative Sharpe over 3+ years (momentum on curve shape stops working)
- Crisis losses exceed -15% in single year (curve dislocations too severe)
- Signal degeneracy: >70% of days in same direction (loss of regime detection)

**Research questions to verify in 2026:**
- Has Fed policy regime changed post-2023, affecting curve momentum?
- Can multiple fly structures (2s5s10s, 3s7s10s) improve diversification?
- What is the optimal momentum lookback (63d vs 42d vs 84d)?
- How does Rank Fly perform during Fed policy pivots (rate cuts/hikes)?
- Can adaptive position sizing reduce drawdowns during curve inversions?

**Needs web research?** Yes

**Web research priority:** Medium (yield curve momentum literature, Fed policy impact on curve)

---

### IDEA-012: Curve RV — Pack Slope Momentum (SR3 Front vs Back)

**Title:** SOFR Pack Slope Momentum (Front vs Back Packs)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 844-849)
- `docs/SOTs/STRATEGY.md` (lines 187-191, 318, 363-369)
- `docs/components/strategies/SR3_CARRY_CURVE.md`
- `src/strategies/rates_curve_rv/sr3_curve_rv_momentum.py`

**Asset Class:** Rates (SOFR futures)

**Strategy Family:** Curve RV / Momentum-Driven Regime

**Description:**
Captures steepening/flattening momentum in yield curve by comparing front pack vs back pack. Pack slope = pack_back - pack_front. Momentum on slope (not mean-reversion). Market-neutral spread construction. Orthogonal to Rank Fly (0.18 signal correlation). Promoted Dec 2025 at 3% weight (Core v9).

**Signals/Drivers:**
- Pack slope: pack_back - pack_front (front vs back curve slope)
- Momentum on slope: sign(slope_today - slope_lagged)
- Execution lag: signal.shift(1) (frozen)

**Historical Implementation:**
- Phase-0: PASS (Sharpe 0.42, momentum variant; mean-reversion failed)
- Phase-1: PASS (Sharpe 0.28, 63d momentum lookback)
- Phase-2: PASS (integrated into Core v9 at 3% weight)
- Currently: 3% weight in Core v9

**Candidate Modern Implementation:**
- Already implemented as canonical (no major changes proposed)
- Optional: Alternative pack definitions (white vs red vs green packs)
- Optional: Adaptive lookback (shorter in high-vol regimes)

**Compatibility:**
✅ **Futures-Six candidate** — Active in Core v9 baseline (3% weight)

**Classification Justification:**
Promoted curve sleeve. ROADMAP.md: "Pack Slope (front vs back) — ✅ PROMOTED" with "Core v9 weight: 3%." STRATEGY.md: "Curve RV Meta-Sleeve (8%: Rank Fly 5% + Pack Slope 3%) [...] both atomics integrated into Core v9."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Correlation with Rank Fly exceeds 0.80 (redundancy within Curve RV meta-sleeve)
- Negative Sharpe over 3+ years (slope momentum stops working)
- Crisis losses exceed -15% in single year (curve dislocations too severe)
- Signal degeneracy: >70% of days in same direction

**Research questions to verify in 2026:**
- What is the correlation with 2s10s Treasury slope (cross-market validation)?
- Can alternative pack definitions improve Sharpe?
- What is the optimal momentum lookback (63d vs 42d vs 84d)?
- How does Pack Slope perform during Fed policy pivots?
- Can this signal be adapted to Treasury curve (ZT/ZF/ZN/UB)?

**Needs web research?** Yes

**Web research priority:** Low (pack slope is standard curve strategy)

---

### IDEA-013: Macro Regime Filter (Vol/Breadth/FRED Overlay)

**Title:** Macro Regime Filter (Volatility/Breadth/FRED Indicators)

**Source Files:**
- `docs/components/strategies/MACRO_REGIME_FILTER.md`
- `src/agents/overlay_macro_regime.py`
- `examples/demo_macro_regime.py`
- `configs/strategies.yaml` (macro_regime section)
- `configs/fred_series.yaml`

**Asset Class:** Overlay (applies to all strategies)

**Strategy Family:** Risk Management / Regime Overlay

**Description:**
Applies continuous scaler k ∈ [0.5, 1.0] to strategy signals based on internal market regime (realized vol, breadth) and external FRED economic indicators. Reduces exposure in adverse conditions (high vol, poor breadth, negative FRED) and increases in favorable conditions. Rebalances weekly (W-FRI). Includes 10 FRED indicators: VIX, DGS10, FEDFUNDS, CPI, UNRATE, etc.

**Signals/Drivers:**
- Realized volatility (21d rolling, ES+NQ equal-weighted)
- Market breadth (fraction above 200d SMA)
- FRED economic indicators (8 daily + 2 monthly, z-scored)
- EMA smoothing (5d half-life for inputs, 15% for final scaler)

**Historical Implementation:**
- Phase-0: Not applicable (overlay, not standalone strategy)
- Phase-1: Implemented and integrated into ExecSim (Nov 2025)
- Phase-2: Active in backtests (CAGR 16.28%, Sharpe 0.94, MaxDD -18.66%)
- Currently: Optional overlay (not always enabled in Core baselines)

**Candidate Modern Implementation:**
- Already implemented as canonical (MacroRegimeFilter)
- Optional: Asset-class specific FRED signals (different indicators for rates vs commodities)
- Optional: Adaptive k_bounds (widen in trending markets, tighten in sideways)

**Compatibility:**
⚠️ **Sidecar / Overlay** — Not a return source; risk management layer

**Classification Justification:**
Active overlay in system. MACRO_REGIME_FILTER.md: "Successfully implemented a complete, production-ready MacroRegimeFilter overlay" with "Integration Status: ✅ Completed" and "Integrated into ExecSim and running in production."

**Extraction Confidence:** High

**What would invalidate this idea?**
- No improvement in MaxDD reduction over 3+ years (overlay not adding value)
- Correlation with vol targeting exceeds 0.95 (redundant with RT layer)
- Crisis losses: overlay fails to reduce exposure in 2020 Q1-style events
- FRED indicators become stale or discontinued (data availability issues)

**Research questions to verify in 2026:**
- Have FRED indicators remained predictive post-2023 (regime change)?
- Can asset-class specific FRED signals improve performance?
- What is the optimal FRED weight (30% vs 20% vs 50%)?
- How does macro filter interact with allocator layer (redundancy check)?
- Can adaptive k_bounds improve Sharpe (regime-dependent exposure limits)?
- Does input smoothing (5d EMA) prevent scaler instability?

**Needs web research?** Yes

**Web research priority:** Medium (FRED indicator predictive power, post-COVID regime)

---

## Parked/Failed Ideas (Phase-0/1/2 Failures)

### IDEA-014: FX/Commodity Carry (Roll Yield Sign-Only)

**Title:** FX and Commodity Roll Yield Carry

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 793-799)
- `docs/SOTs/STRATEGY.md` (lines 478-481)
- `carry_phase0_run_memo.md`

**Asset Class:** FX (6E, 6B, 6J), Commodities (CL, GC)

**Strategy Family:** Carry / Roll Yield

**Description:**
Captures roll yield from futures term structure (front vs rank 1 contracts). Long when contango (positive roll yield), short when backwardation (negative roll yield). Sign-only Phase-0 implementation. Failed with negative Sharpe (-0.69) over 2020-2025 window. Simple roll yield strategy showed negative alpha in recent years.

**Signals/Drivers:**
- Roll yield: (Rank_1_price - Front_price) / Front_price
- Sign-only signal: Long if roll yield positive, short if negative

**Historical Implementation:**
- Phase-0: FAIL (Sharpe -0.69, negative across all assets)
- Status: PARKED for redesign

**Candidate Modern Implementation:**
- Sector-based roll yield (separate logic for FX vs commodities vs rates)
- DV01-neutral carry for rates (not just sign-only)
- Regime-dependent filters (USD strength for FX, commodity cycles for CL/GC)
- Cross-sectional ranking within asset classes

**Compatibility:**
❌ **Likely obsolete or superseded** — Failed Phase-0, parked for redesign

**Classification Justification:**
Explicitly failed Phase-0. ROADMAP.md: "FX/Commodity Carry: Parked for redesign — Phase-0 Sanity Check Result: Negative Sharpe (-0.69) across all assets (2020-2025)." Carry_phase0_run_memo.md confirms Phase-0 failure.

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Sharpe -0.69 in Phase-0
- Roll yield sign no longer predictive in post-2010 markets
- Commodities: backwardation dominates (inventory/storage dynamics changed)
- FX: carry trade unwind risk dominates roll yield (USD flight-to-safety)

**Research questions to verify in 2026:**
- Has commodity term structure changed post-2020 (pandemic/war impact)?
- Can sector-based logic rescue carry (e.g., only trade CL in contango)?
- Is DV01-neutral rates carry feasible with current data?
- Can cross-sectional ranking (best carry vs worst) improve performance?
- What is the correlation between roll yield and momentum in commodities?

**Needs web research?** Yes

**Web research priority:** High (commodity term structure, FX carry literature)

---

### IDEA-015: Equity Carry (Implied Dividend Yield)

**Title:** Equity Index Implied Dividend Carry

**Source Files:**
- `equity_carry_forensic_memo.md`
- `carry_equity_carry_fix_memo.md`
- `carry_phase0_run_memo.md`
- `src/agents/feature_equity_carry.py`

**Asset Class:** Equity indices (ES, NQ, RTY)

**Strategy Family:** Carry / Implied Dividend

**Description:**
Captures implied dividend yield from equity index futures vs spot. Formula: d_implied = r - (1/T) * log(F/S). Long when implied dividend yield is attractive (high). Individual signals strong (ES Sharpe 1.57, NQ 2.25, RTY 1.09), but ensemble Sharpe -0.54. Implied dividend calculation produces impossible values (ES mean -70%, NQ mean -1149%).

**Signals/Drivers:**
- Implied dividend yield: d_implied = SOFR - (1/T) * log(Futures/Spot)
- T = 45 days (constant approximation for front-month futures)
- Signal: Long when d_implied > threshold

**Historical Implementation:**
- Phase-0: Individual signals strong, ensemble weak
- Phase-1: Non-admissible (ensemble Sharpe -0.54)
- Status: PARKED — excluded from Carry Meta v1

**Candidate Modern Implementation:**
- Fix implied dividend calculation (investigate formula/data errors)
- Use actual expiry dates (not constant T = 45 days)
- Verify spot indices are price-return (not total return)
- Treat as policy feature (Layer 2 overlay), not return sleeve

**Compatibility:**
❌ **Likely obsolete or superseded** — Non-admissible as Engine v1 return source

**Classification Justification:**
Explicitly excluded from Carry Meta v1. equity_carry_forensic_memo.md: "CONCLUSION: Equity carry is NON-ADMISSIBLE as an Engine v1 return source." Recommendation: "Exclude Equity from Carry Meta v1" and "Treat Equity Implied Dividends as Policy Feature."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Ensemble Sharpe -0.54 despite strong individual signals
- Implied dividend calculation broken (impossible values: ES -70%, NQ -1149%)
- Data quality issues (stale spot prices, wrong futures contract month)
- Formula error or unit conversion error (percentage vs decimal)

**Research questions to verify in 2026:**
- What is the root cause of impossible implied dividend values?
- Are spot indices price-return or total return (FRED documentation check)?
- Does using actual expiry dates (not T=45) fix the calculation?
- Can equity implied dividends work as a policy overlay (not return sleeve)?
- How do implied dividends correlate with equity momentum signals?

**Needs web research?** Yes

**Web research priority:** Medium (equity futures pricing, implied dividend mechanics)

---

### IDEA-016: VRP-Convergence (VIX - VX1)

**Title:** Volatility Risk Premium — Convergence (VIX vs VX1)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 64-66, 730-746)
- `docs/SOTs/STRATEGY.md` (lines 280-281, 372-386)
- `reports/phase4_research/vrp_autopsy/VRP_PHASE4_DECISION.md`

**Asset Class:** Volatility (VIX futures: VX1)

**Strategy Family:** Volatility Risk Premium / Convergence Trade

**Description:**
Short-only strategy targeting convergence between VIX and VX1. Short VX1 when VIX < VX1 (backwardation/convergence opportunity). Z-scored over 252d window. Phase-2 integrated into Core v6/v7/v8/v9 but de-emphasized (2.5% → 2.185% weight). Phase 4 audit revealed negative unconditional contribution (Sharpe -0.253). PARKED after Phase 4 (Jan 2026).

**Signals/Drivers:**
- Convergence spread: VIX - VX1 (negative when backwardation)
- Z-scored over 252d rolling window
- Signal: Short VX1 when z-score negative (VIX < VX1)

**Historical Implementation:**
- Phase-0: PASS (short-only signal test)
- Phase-1: PASS (engineered sleeve with z-scoring)
- Phase-2: PASS (integrated into Core v6, promoted Dec 2025)
- Phase 4: FAIL (negative contribution Sharpe -0.253, removed from VRP_v2)
- Status: PARKED (tested-negative, archived Jan 2026)

**Candidate Modern Implementation:**
- Not recommended (Phase 4 audit concluded it's a belief defect)
- Alternative: Use VIX-VX1 spread as policy feature (not return source)

**Compatibility:**
❌ **Likely obsolete or superseded** — PARKED after Phase 4 audit (Jan 2026)

**Classification Justification:**
Explicitly parked after Phase 4. VRP_PHASE4_DECISION.md: "VRP-Convergence is a proven negative belief component" with "Contribution Sharpe: -0.253" and "Status: 🗄️ PARKED — Tested → Negative → Archived." ROADMAP.md: "VRP-Convergence economically parked (2.5% weight, kept for historical continuity only)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Phase 4 audit showed -0.253 contribution Sharpe
- Negative unconditional contribution to VRP meta-sleeve
- VIX-VX1 backwardation no longer predictive of profitable shorts
- Correlation with VRP-Core too high (redundant)

**Research questions to verify in 2026:**
- Has VIX-VX1 spread behavior changed post-2023 (term structure regime)?
- Can VIX-VX1 spread improve performance as a policy overlay (not return)?
- What is the correlation with VRP-Core in different vol regimes?
- Does convergence signal work better as a crisis filter (reduce VRP-Core exposure)?

**Needs web research?** No

**Web research priority:** N/A (already thoroughly tested and rejected)

---

### IDEA-017: VRP-RollYield (VX1 Roll-Down Carry)

**Title:** VRP Roll Yield (VX1 per-day-to-expiry roll-down)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 687-703)

**Asset Class:** Volatility (VIX futures: VX1)

**Strategy Family:** Volatility Risk Premium / Roll Yield

**Description:**
Captures front-month roll-down carry by shorting VX1 when future trades above VIX on per-day-to-expiry basis. Short VX1 when roll_yield = (VX1 - VIX) / days_to_expiry > 0. Phase-0 result: Sharpe +0.02 (below 0.10 threshold), MaxDD -85.65%. Borderline, near-zero economic edge with unacceptable path profile. PARKED.

**Signals/Drivers:**
- Roll yield: (VX1 - VIX) / days_to_expiry
- Signal: Short VX1 when roll yield positive

**Historical Implementation:**
- Phase-0: FAIL (Sharpe 0.02, below 0.10 threshold; MaxDD -85.65%)
- Status: PARKED (borderline, near-zero edge)

**Candidate Modern Implementation:**
- Not recommended (Phase-0 showed near-zero edge)
- Alternative: Multi-tenor basket (VX1/VX2/VX3 weighted by roll yield)
- Alternative: Integrated component (not standalone sleeve)

**Compatibility:**
❌ **Likely obsolete or superseded** — PARKED after borderline Phase-0

**Classification Justification:**
Explicitly parked. ROADMAP.md: "VRP-RollYield (Status: PARKED — Borderline Phase-0)" with "Sharpe +0.02 (below the ≥ 0.10 Phase-0 bar)" and "MaxDD –85.65% (slightly worse than the catastrophic threshold)." Decision: "PARKED."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Sharpe 0.02 (near-zero), MaxDD -85.65% (catastrophic)
- Roll-down carry in VX1 no longer exists (efficient pricing)
- Days-to-expiry approximation not accurate enough
- Simple sign rule loses too much information

**Research questions to verify in 2026:**
- Can multi-tenor basket (VX1/VX2/VX3) improve edge?
- Is roll yield calculation correct (days-to-expiry vs actual calendar)?
- Can roll yield work as policy feature (overlay on VRP-Core)?

**Needs web research?** No

**Web research priority:** N/A (Phase-0 failure clear)

---

### IDEA-018: VRP-TermStructure (VX Slope Directional)

**Title:** VRP Term Structure Slope (VX2-VX1 Directional)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 705-720)

**Asset Class:** Volatility (VIX futures: VX1)

**Strategy Family:** Volatility Risk Premium / Term Structure

**Description:**
Directional VX1 trading based on VX2-VX1 slope. Short VX1 when slope steep (contango). Phase-0 confirmed VX2-VX1 slope is not a directional VRP signal. Does not map cleanly to outright front-month selling. High overlap with VIX-VX1 but weaker economics. PARKED. Better used as regime feature, not return engine.

**Signals/Drivers:**
- VX2 - VX1 slope (term structure steepness)
- Signal: Short VX1 when slope positive (contango)

**Historical Implementation:**
- Phase-0: FAIL (confirmed via Phase-0: not a directional VRP signal)
- Status: PARKED (better as regime feature)

**Candidate Modern Implementation:**
- Not recommended as return engine
- Alternative: Regime filter for VRP-Core/Convergence (reduce exposure when slope inverts)
- Alternative: Crisis/Long-Vol Meta-Sleeve component

**Compatibility:**
❌ **Likely obsolete or superseded** — PARKED after Phase-0 failure

**Classification Justification:**
Explicitly parked. ROADMAP.md: "VRP-TermStructure (slope-only directional trades) — Status: PARKED" with "Confirmed via Phase-0: VX2 − VX1 slope is not a directional VRP signal" and "Better used as regime feature → not return engine."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Phase-0 confirmed not a directional signal
- VX slope does not predict profitable VX1 shorts
- Overlap with VIX-VX1 too high (redundant)

**Research questions to verify in 2026:**
- Can VX slope improve VRP-Core as a regime filter (not return)?
- Is there predictive power in slope changes (not levels)?
- Can VX slope work in calendar-spread format (VX2-VX1 spread trade)?

**Needs web research?** No

**Web research priority:** N/A (Phase-0 failure clear, better as regime feature)

---

### IDEA-019: VRP-FrontSpread (VX1 > VX2 Directional)

**Title:** VRP Front Spread Richness (VX1 > VX2 Directional Short)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 541-558)

**Asset Class:** Volatility (VIX futures: VX1)

**Strategy Family:** Volatility Risk Premium / Calendar Richness

**Description:**
Short VX1 when VX1 > VX2 (backwardation/contango inversion). Calendar-richness does not map to profitable outright VX1 short. VX term structure usually in backwardation (VX1 < VX2), so contango signal only triggers in small subset of regimes. Simple directional short not right expression of "calendar carry." PARKED after Phase-0 failure.

**Signals/Drivers:**
- VX1 - VX2 spread (calendar richness)
- Signal: Short VX1 when VX1 > VX2

**Historical Implementation:**
- Phase-0: FAIL (calendar-richness does not map to profitable VX1 short)
- Status: PARKED (revisit only as calendar-spread trade or feature/regime input)

**Candidate Modern Implementation:**
- Not recommended as directional VX1 trade
- Alternative: Calendar-spread trade (short VX1-VX2 spread, not outright VX1)
- Alternative: Feature/regime input (not return engine)

**Compatibility:**
❌ **Likely obsolete or superseded** — PARKED after Phase-0 failure

**Classification Justification:**
Explicitly parked. ROADMAP.md: "VRP-FrontSpread (Directional) ❌ PARKED" with "Calendar-richness does not map to profitable outright VX1 short" and "Status: ❌ PARKED — Phase-0 FAIL. Revisit only as calendar-spread trade or feature/regime input."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Phase-0 failure (directional VX1 short not profitable)
- Backwardation is structural (VX1 < VX2 most of the time)
- Contango signal too rare (small subset of regimes)

**Research questions to verify in 2026:**
- Can calendar-spread trade (short VX1-VX2 spread) work?
- Is VX1 > VX2 signal useful as regime feature (not return)?
- What % of days is VX term structure in contango (post-2023)?

**Needs web research?** No

**Web research priority:** N/A (Phase-0 failure clear)

---

### IDEA-020: VRP-Convexity Premium (VVIX Threshold Short)

**Title:** VRP Convexity Premium (VVIX > 100 Short VX1)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 515-540)

**Asset Class:** Volatility (VIX futures: VX1)

**Strategy Family:** Volatility Risk Premium / Vol-of-Vol

**Description:**
Short VX1 when VVIX > 100 (elevated vol-of-vol indicates expensive convexity insurance). Idea: Once shocks stabilize, short-vol carry favorable. Phase-0 tested threshold-based directional short. Failed. VVIX data now available but expression failed. PARKED. Revisit only as conditioning feature or spread-style trade, not directional.

**Signals/Drivers:**
- VVIX (CBOE vol-of-vol index)
- Signal: Short VX1 when VVIX > 100

**Historical Implementation:**
- Phase-0: FAIL (threshold short VX1 not profitable)
- Status: PARKED (VVIX should be conditioning feature for Phase-B / Crisis, not engine)

**Candidate Modern Implementation:**
- Not recommended as directional short
- Alternative: Conditioning feature for VRP-Core (reduce exposure when VVIX elevated)
- Alternative: Crisis filter (de-risk when VVIX spikes)

**Compatibility:**
❌ **Likely obsolete or superseded** — PARKED after Phase-0 failure

**Classification Justification:**
Explicitly parked. ROADMAP.md: "VRP-Convexity Premium (VVIX Relative Value) — Status: PARKED — Phase-0 FAIL (threshold short VX1)" with "VVIX data now available; expression failed; revisit only as conditioning feature or spread-style trade."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Phase-0 failure (threshold short not profitable)
- VVIX > 100 not predictive of profitable VX1 shorts
- Simple threshold rule loses too much information

**Research questions to verify in 2026:**
- Can VVIX improve VRP-Core as conditioning feature (not return)?
- Is VVIX predictive of vol spikes (forward-looking)?
- Can VVIX-based crisis filter reduce VRP drawdowns?

**Needs web research?** Yes

**Web research priority:** Medium (VVIX mechanics, institutional usage)

---

### IDEA-021: VRP-Structural (VIX - RV252)

**Title:** VRP Structural Long-Horizon (VIX vs RV252)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 563-585)

**Asset Class:** Volatility (VIX futures: VX1, VX2, VX3)

**Strategy Family:** Volatility Risk Premium / Long-Horizon

**Description:**
Long-horizon implied vs realized vol premium: VIX (1-month implied) vs RV252 (252-day realized vol). Short VX when VIX > RV252. Tested three variants: VX1, VX2, VX3 (all same signal logic). All failed Phase-0 (Sharpe < 0.10). VIX > RV252 occurs ~75% of time but shorting VX not profitable. Exposes to crisis convexity risk. PARKED.

**Signals/Drivers:**
- VIX - RV252 spread (long-horizon implied vs realized)
- Signal: Short VX when VIX > RV252

**Historical Implementation:**
- Phase-0: FAIL (all three variants: VX1, VX2, VX3 failed Sharpe < 0.10)
- Status: PARKED (not an engine in simple form; revisit as conditioning or different instrument)

**Candidate Modern Implementation:**
- Not recommended as directional short
- Alternative: Conditioning feature (reduce VRP-Core exposure when spread extreme)
- Alternative: Different instrument/expression (options, not futures)

**Compatibility:**
❌ **Likely obsolete or superseded** — PARKED after Phase-0 failure

**Classification Justification:**
Explicitly parked. ROADMAP.md: "VRP-Structural (RV252) ❌ PARKED — Phase-0 FAIL (VX1/VX2/VX3)" with "Decision: All three variants failed Phase-0 criteria (Sharpe < 0.10)." Status: "❌ PARKED — Phase-0 FAIL. Not an engine in simple form."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Phase-0 failure across all variants (Sharpe < 0.10)
- VIX > RV252 occurs 75% of time (signal degeneracy)
- Shorting VX in these regimes not profitable (crisis convexity risk)

**Research questions to verify in 2026:**
- Can long-horizon spread work as conditioning feature (not return)?
- Is there predictive power in spread changes (not levels)?
- Can options-based expression work (not VX futures)?

**Needs web research?** No

**Web research priority:** N/A (Phase-0 failure clear across all variants)

---

### IDEA-022: VRP-Mid (VIX - RV126)

**Title:** VRP Mid-Horizon (VIX vs RV126)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 587-609)

**Asset Class:** Volatility (VIX futures: VX2, VX3)

**Strategy Family:** Volatility Risk Premium / Mid-Horizon

**Description:**
Mid-horizon implied vs realized vol premium: VIX (1-month implied) vs RV126 (126-day realized vol). Short VX when VIX > RV126. Tested two variants: VX2, VX3 (back-month futures only). Both failed Phase-0 (Sharpe < 0.10). VIX > RV126 occurs ~82% of time but shorting VX not profitable. Crisis convexity risk. PARKED.

**Signals/Drivers:**
- VIX - RV126 spread (mid-horizon implied vs realized)
- Signal: Short VX when VIX > RV126

**Historical Implementation:**
- Phase-0: FAIL (both variants: VX2, VX3 failed Sharpe < 0.10)
- Status: PARKED (not an engine in simple form; revisit as conditioning or different instrument)

**Candidate Modern Implementation:**
- Not recommended as directional short
- Alternative: Conditioning feature (reduce VRP exposure when spread extreme)
- Alternative: Different instrument/expression (options, not futures)

**Compatibility:**
❌ **Likely obsolete or superseded** — PARKED after Phase-0 failure

**Classification Justification:**
Explicitly parked. ROADMAP.md: "VRP-Mid (RV126) ❌ PARKED — Phase-0 FAIL (VX2/VX3)" with "Decision: Both variants failed Phase-0 criteria (Sharpe < 0.10)." Status: "❌ PARKED — Phase-0 FAIL. Not an engine in simple form."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Phase-0 failure across both variants (Sharpe < 0.10)
- VIX > RV126 occurs 82% of time (signal degeneracy)
- Shorting VX in these regimes not profitable (crisis convexity risk)

**Research questions to verify in 2026:**
- Can mid-horizon spread work as conditioning feature (not return)?
- Is there difference between RV126 vs RV252 (structural or just lookback)?
- Can options-based expression work (not VX futures)?

**Needs web research?** No

**Web research priority:** N/A (Phase-0 failure clear, similar to VRP-Structural)

---

### IDEA-023: Crisis Meta-Sleeve (Long VX3)

**Title:** Crisis Tail Risk Meta-Sleeve (Long VX3 Always-On)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 900-953)
- `docs/SOTs/STRATEGY.md` (lines 203-226)

**Asset Class:** Volatility (VIX futures: VX3)

**Strategy Family:** Crisis / Tail Risk Hedge

**Description:**
Always-on convexity hedge via long VX3 exposure. Not expected to produce positive unconditional Sharpe (unlike alpha sleeves). Goal: Improve conditional performance during market stress (equity crashes, vol spikes, macro dislocations). Phase-0/1: VX3 chosen as cheapest usable convexity. Phase-2: FAIL due to 2020 Q1 fast-crash deterioration. NO PROMOTION (Dec 2025).

**Signals/Drivers:**
- None (always-on long exposure)
- No timing or regime logic in v1
- Evaluated on tail behavior, not average returns

**Historical Implementation:**
- Phase-0: PASS (MaxDD +0.91%, Worst-month +0.49%)
- Phase-1: VX3 PROMOTED to Phase-2 (MaxDD +0.41%, Worst-month +0.23%)
- Phase-2: FAIL (overall tail metrics improved but 2020 Q1 fast-crash deteriorated)
- Status: NO PROMOTION (Dec 2025)

**Candidate Modern Implementation:**
- Not recommended for v1 (always-on convexity failed Phase-2)
- Alternative: Conditional activation (allocator-driven, not always-on)
- Alternative: VX2/VX3 blend rules (dynamic convexity allocation)
- Alternative: Options-based convexity (deferred, high complexity)

**Compatibility:**
⚠️ **Sidecar / Allocator-era research** — Not standalone engine; belongs in allocator logic

**Classification Justification:**
Explicitly rejected for v1. ROADMAP.md: "Crisis / Tail Risk Meta-Sleeve (v1 COMPLETE — NO PROMOTION)" with "Final Decision: Crisis Meta-Sleeve v1 — NO PROMOTION" and "Long VX3 fails Phase-2 due to deterioration in 2020 Q1 fast-crash behavior." Disposition: "Crisis protection deferred to allocator logic (v2+)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Phase-2 failure (2020 Q1 fast-crash deterioration)
- Always-on convexity too expensive (negative Sharpe drag)
- 2020 Q1 fast-crash: portfolio-level drawdown worsened despite VX3 gains
- Instrument-level VX behavior correct but portfolio-level integration fails

**Research questions to verify in 2026:**
- Can conditional activation (not always-on) improve Phase-2 results?
- Is there optimal VX2/VX3 blend for tail hedging?
- Can allocator-driven convexity (regime-dependent) reduce cost?
- How does always-on VX3 perform in 2022 drawdown (not just 2020)?
- Can options-based convexity (VIX calls) work better than VX futures?

**Needs web research?** Yes

**Web research priority:** Medium (tail hedging literature, institutional crisis protection)

---

### IDEA-024: Long VX2 (Always-On Benchmark)

**Title:** Crisis Hedge — Long VX2 Always-On

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 910-913, 931-932)

**Asset Class:** Volatility (VIX futures: VX2)

**Strategy Family:** Crisis / Tail Risk Hedge

**Description:**
Always-on long VX2 exposure. Phase-0: PASS (MaxDD +0.91%, Worst-month +0.49%). Not promoted to Phase-1 (used as benchmark ceiling reference only). Retained for comparison but not pursued as production candidate. More expensive than VX3 for same tail protection.

**Signals/Drivers:**
- None (always-on long exposure)
- No timing or regime logic

**Historical Implementation:**
- Phase-0: PASS (MaxDD +0.91%, Worst-month +0.49%)
- Phase-1: Not promoted (benchmark only)
- Status: Benchmark ceiling reference (not production candidate)

**Candidate Modern Implementation:**
- Not recommended (VX3 is cheaper for same tail protection)
- Retained as benchmark only

**Compatibility:**
❌ **Likely obsolete or superseded** — Benchmark only, not production candidate

**Classification Justification:**
Explicitly not promoted. ROADMAP.md: "Long VX2: PASS (MaxDD +0.91%, Worst-month +0.49%)" with "Disposition: VX2: Retained as benchmark only (not promoted)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Not applicable (benchmark only, never intended for production)

**Research questions to verify in 2026:**
- N/A (benchmark only)

**Needs web research?** No

**Web research priority:** N/A

---

### IDEA-025: Long VX2-VX1 Spread (Always-On)

**Title:** Crisis Hedge — Long VX2-VX1 Spread

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 910-913, 931-934)

**Asset Class:** Volatility (VIX futures: VX2-VX1 spread)

**Strategy Family:** Crisis / Tail Risk Hedge

**Description:**
Always-on long VX2-VX1 spread exposure. Phase-0: PASS (MaxDD +0.76%, Worst-month +0.41%). Phase-1: FAIL (insufficient tail preservation). Spread-based hedge does not provide enough convexity. PARKED.

**Signals/Drivers:**
- None (always-on long spread exposure)
- No timing or regime logic

**Historical Implementation:**
- Phase-0: PASS (MaxDD +0.76%, Worst-month +0.41%)
- Phase-1: FAIL (insufficient tail preservation)
- Status: PARKED

**Candidate Modern Implementation:**
- Not recommended (insufficient tail preservation)

**Compatibility:**
❌ **Likely obsolete or superseded** — PARKED after Phase-1 failure

**Classification Justification:**
Explicitly parked. ROADMAP.md: "Long VX2 - VX1 spread: ❌ PARKED (Phase-1 FAIL — insufficient tail preservation)." Disposition: "VX Spreads: Parked (Phase-1 FAIL)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Phase-1 failure (insufficient tail preservation)
- Spread does not provide enough convexity in crisis

**Research questions to verify in 2026:**
- N/A (Phase-1 failure clear)

**Needs web research?** No

**Web research priority:** N/A

---

### IDEA-026: Long VX3-VX2 Spread (Always-On)

**Title:** Crisis Hedge — Long VX3-VX2 Spread

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 917-918, 931-934)

**Asset Class:** Volatility (VIX futures: VX3-VX2 spread)

**Strategy Family:** Crisis / Tail Risk Hedge

**Description:**
Always-on long VX3-VX2 spread exposure. Phase-1: FAIL (insufficient tail preservation). Spread-based hedge does not provide enough convexity. PARKED.

**Signals/Drivers:**
- None (always-on long spread exposure)
- No timing or regime logic

**Historical Implementation:**
- Phase-1: FAIL (insufficient tail preservation)
- Status: PARKED

**Candidate Modern Implementation:**
- Not recommended (insufficient tail preservation)

**Compatibility:**
❌ **Likely obsolete or superseded** — PARKED after Phase-1 failure

**Classification Justification:**
Explicitly parked. ROADMAP.md: "Long VX3 - VX2 spread: ❌ PARKED (Phase-1 FAIL — insufficient tail preservation)." Disposition: "VX Spreads: Parked (Phase-1 FAIL)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Already invalidated: Phase-1 failure (insufficient tail preservation)
- Spread does not provide enough convexity in crisis

**Research questions to verify in 2026:**
- N/A (Phase-1 failure clear)

**Needs web research?** No

**Web research priority:** N/A

---

### IDEA-027: Long UB (Conditional Rates Hedge)

**Title:** Crisis Hedge — Long UB (30-Year Treasury)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 912-913, 931-933)

**Asset Class:** Rates (UB: 30-Year Treasury futures)

**Strategy Family:** Crisis / Flight-to-Quality Hedge

**Description:**
Conditional hedge via long 30-Year Treasury exposure. Phase-0: PASS (parked for post-v1 reconsideration). Not pursued for v1 (conditional hedge logic belongs in allocator era). Retained for future consideration.

**Signals/Drivers:**
- None specified (conditional hedge logic not defined)
- Future: Likely regime-dependent activation (not always-on)

**Historical Implementation:**
- Phase-0: PASS (parked for post-v1 reconsideration)
- Status: Parked (conditional hedge, post-v1 reconsideration)

**Candidate Modern Implementation:**
- Not recommended for v1 (conditional logic belongs in allocator)
- Future: Allocator-driven long UB in crisis regimes (VIX > 30, equity drawdown > -10%)

**Compatibility:**
⚠️ **Sidecar / Allocator-era research** — Conditional hedge belongs in allocator logic

**Classification Justification:**
Explicitly parked for post-v1. ROADMAP.md: "Long UB: PASS (parked for post-v1 reconsideration)." Disposition: "UB: Remains parked (conditional hedge, post-v1 reconsideration)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Flight-to-quality no longer predictable (rates regime change)
- Negative correlation with equities breaks down (post-2020 example: 2022 both down)
- Transaction costs too high (frequent switching in/out)

**Research questions to verify in 2026:**
- Does flight-to-quality still work in current rates regime (post-2023)?
- Can VIX/equity drawdown trigger long UB exposure effectively?
- What is the optimal entry/exit logic for conditional UB hedge?
- How did UB perform in 2022 drawdown (rates and equities both down)?

**Needs web research?** Yes

**Web research priority:** Low (conditional hedging belongs in allocator era, not v1)

---

## Planned/Future Ideas (Not Yet Implemented)

### IDEA-028: VRP-CalendarDecay (Tenor Ladder Decay)

**Title:** VRP Calendar Decay (Tenor-Specific Decay)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 615-637)

**Asset Class:** Volatility (VIX futures: VX2, VX3)

**Strategy Family:** Volatility Risk Premium / Calendar Decay

**Description:**
Each VX future decays toward its "anchor": VX1 → VIX, VX2 → VX1, VX3 → VX2. Short VX2 if VX2 > VX1, short VX3 if VX3 > VX2. Basket-weighted (tenor decay sleeve). More robust than VRP-RollYield (which relied on VX1 alone). Closer to how actual volatility desks trade VRP. Not yet implemented, medium priority for Phase-0 testing.

**Signals/Drivers:**
- VX2 > VX1 → short VX2 (decay toward anchor)
- VX3 > VX2 → short VX3 (decay toward anchor)
- Basket-weighted or individual sleeve per tenor

**Historical Implementation:**
- Not yet implemented
- Status: Planned (medium priority for Phase-0 testing)

**Candidate Modern Implementation:**
- Phase-0: Sign-only tenor decay signals
- Phase-1: Z-scoring, vol targeting, basket weighting
- Phase-2: Integration into VRP meta-sleeve (if passes Phase-0/1)

**Compatibility:**
✅ **Futures-Six candidate** — Not yet tested, medium priority

**Classification Justification:**
Explicitly planned. ROADMAP.md: "VRP Sleeve #5 — VRP-CalendarDecay (Tenor Ladder Decay)" with "Status: Not yet implemented. Medium priority for Phase-0 testing."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Phase-0 failure: Sharpe < 0.10
- Correlation with VRP-Core exceeds 0.95 (redundancy)
- VX term structure inverts too frequently (decay signal degenerates)
- Transaction costs exceed alpha (multiple tenor trades)

**Research questions to verify in 2026:**
- Is tenor decay (VX2→VX1, VX3→VX2) more robust than simple VRP-Core?
- Can basket weighting improve Sharpe vs individual tenor sleeves?
- What is the optimal signal construction (sign-only vs z-score)?
- How does tenor decay perform in backwardation regimes (VX term structure inverted)?
- Can this replace VRP-RollYield (which failed Phase-0)?

**Needs web research?** Yes

**Web research priority:** Medium (institutional vol trading practices, tenor decay mechanics)

---

### IDEA-029: VRP-ReverseConvergence (Crisis Long-Vol)

**Title:** VRP Reverse Convergence (Crisis Long-Vol Sleeve)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 665-682)

**Asset Class:** Volatility (VIX futures: VX1)

**Strategy Family:** Volatility / Crisis / Long-Vol

**Description:**
Long-vol sleeve for VRP stabilization. When backwardation gets extreme (VIX - VX1 > 3), go long VX1. Captures VRP carry unwind during violent regime shifts. Can be: (a) Crisis Meta-Sleeve component, (b) VRP shock absorber, or (c) exposure dial for VRP-Core. Phase-0 should show positive Sharpe (long-vol during extreme backwardation often works). Not yet implemented, lower priority (may belong in Crisis Meta-Sleeve).

**Signals/Drivers:**
- VIX - VX1 spread (backwardation severity)
- Signal: Long VX1 when VIX - VX1 > 3 (extreme backwardation)

**Historical Implementation:**
- Not yet implemented
- Status: Planned (lower priority, may belong in Crisis Meta-Sleeve)

**Candidate Modern Implementation:**
- Phase-0: Sign-only long VX1 when extreme backwardation
- Phase-1: Threshold optimization (is 3 optimal, or 2 or 4?)
- Phase-2: Integration as Crisis component or VRP shock absorber

**Compatibility:**
⚠️ **Sidecar / Crisis Meta-Sleeve** — Not core VRP sleeve; belongs in crisis/defensive logic

**Classification Justification:**
Explicitly planned but lower priority. ROADMAP.md: "VRP Sleeve #7 — VRP-ReverseConvergence (Crisis Turning Point Sleeve)" with "Status: Not yet implemented. Lower priority (may belong in Crisis Meta-Sleeve)."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Phase-0 failure: Sharpe < 0.0 (long-vol during extreme backwardation unprofitable)
- Extreme backwardation too rare (signal triggers <5% of days)
- Transaction costs exceed alpha (timing too difficult)
- Correlation with Crisis Meta-Sleeve (Long VX3) exceeds 0.95 (redundancy)

**Research questions to verify in 2026:**
- Is VIX - VX1 > 3 optimal threshold, or should it be 2 or 4?
- How often does extreme backwardation occur (2020-2026)?
- Can this sleeve reduce VRP drawdowns during vol spikes?
- Should this be Crisis Meta-Sleeve component or VRP shock absorber?
- How does this correlate with Long VX3 (Crisis sleeve that failed Phase-2)?

**Needs web research?** Yes

**Web research priority:** Medium (crisis turning points, long-vol mechanics)

---

### IDEA-030: VIX Curve Curvature (Fly)

**Title:** VIX Curve Curvature (Calendar Fly)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 639-663)

**Asset Class:** Volatility (VIX futures: VX1, VX2, VX3)

**Strategy Family:** Volatility / Curve Shape

**Description:**
Calendar fly construction: 2*VX2 - VX1 - VX3. Tests whether VIX term-structure curvature contains incremental information beyond VX2-VX1 carry. Liquidity beyond VX2 limited. Expected redundancy with VX calendar carry. Included for research completeness; promotion unlikely unless behavior clearly distinct. Not tested in v1, deferred to post-v1 expansion cycle (v2), low priority.

**Signals/Drivers:**
- Fly: 2*VX2 - VX1 - VX3 (curvature in rate space)
- Signal: Trade fly based on curvature signal (specifics TBD)

**Historical Implementation:**
- Not yet implemented
- Status: Planned (deferred to v2, low priority)

**Candidate Modern Implementation:**
- Phase-0: Sign-only fly signals (long when curvature positive?)
- Phase-1: Z-scoring, vol targeting, fly construction rules
- Phase-2: Integration if passes Phase-0/1 (unlikely due to expected redundancy)

**Compatibility:**
⚠️ **Likely obsolete or superseded** — Low priority, expected redundancy with VX carry

**Classification Justification:**
Explicitly low priority. ROADMAP.md: "VRP Sleeve #6 — VIX Curve Curvature (Fly) — PLANNED (v2, Low Priority)" with "Expected redundancy with VX calendar carry" and "Promotion unlikely unless behavior is clearly distinct."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Phase-0 failure: Sharpe < 0.10
- Correlation with VX2-VX1 carry exceeds 0.90 (expected redundancy confirmed)
- Liquidity issues (VX3 bid-ask spreads too wide)
- Fly construction too complex for marginal benefit

**Research questions to verify in 2026:**
- Does VIX curvature contain information beyond slope (VX2-VX1)?
- What is the correlation between fly and VX calendar carry?
- Can fly construction work with limited VX3 liquidity?
- Is there predictive power in curvature changes (not levels)?

**Needs web research?** Yes

**Web research priority:** Low (expected redundancy, low priority)

---

### IDEA-031: Treasury Curve RV (ZT/ZF/ZN/UB Momentum)

**Title:** Treasury Curve Relative Value — Momentum

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 872-889)

**Asset Class:** Rates (Treasuries: ZT, ZF, ZN, UB)

**Strategy Family:** Curve RV / Momentum-Driven Regime

**Description:**
Apply curve RV momentum logic to Treasury curve. Instruments: ZT (2y), ZF (5y), ZN (10y), UB (30y). Planned expressions: 2s10s slope momentum, 5s30s slope momentum, 2s5s10s fly momentum, 5s10s30s fly momentum. Requires DV01-aware construction (different contracts have different DV01s). Phase-0 to be conducted post-v1 freeze. Deferred to v2 expansion cycle.

**Signals/Drivers:**
- 2s10s slope momentum (ZT-ZN spread momentum)
- 5s30s slope momentum (ZF-UB spread momentum)
- 2s5s10s fly momentum (fly construction)
- 5s10s30s fly momentum (fly construction)

**Historical Implementation:**
- Not yet implemented
- Status: Planned (deferred to v2 expansion cycle)

**Candidate Modern Implementation:**
- Phase-0: Sign-only momentum on Treasury slopes/flies
- Phase-1: DV01-aware construction, z-scoring, vol targeting
- Phase-2: Integration into Curve RV meta-sleeve (if passes Phase-0/1)

**Compatibility:**
✅ **Futures-Six candidate** — Not yet tested, deferred to v2

**Classification Justification:**
Explicitly planned for v2. ROADMAP.md: "v2 (PLANNED) — Treasury Curve RV" with "Requirements: DV01-aware construction" and "Status: Deferred to v2 expansion cycle."

**Extraction Confidence:** High

**What would invalidate this idea?**
- Phase-0 failure: Sharpe < 0.10
- Correlation with SR3 Curve RV exceeds 0.90 (redundancy across SOFR/Treasury)
- DV01 adjustments too complex for marginal benefit
- Treasury liquidity deteriorates (bid-ask spreads widen)

**Research questions to verify in 2026:**
- Does Treasury curve momentum work like SOFR curve momentum?
- What is the correlation between SOFR and Treasury curve signals?
- Can DV01-aware construction improve Sharpe vs naive slope?
- How do 2s10s and 5s30s slopes differ in predictive power?
- Can Treasury fly constructions add incremental information?

**Needs web research?** Yes

**Web research priority:** Medium (Treasury curve dynamics, DV01 construction)

---

### IDEA-032: Cross-Market Basis RV (SOFR ↔ Treasuries)

**Title:** Cross-Market Basis Relative Value (SOFR vs Treasuries)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 890-893)

**Asset Class:** Rates (SOFR futures vs Treasuries)

**Strategy Family:** Macro RV / Basis Trading

**Description:**
Capture macro RV between SOFR and Treasury curves. Not pure Curve RV (cross-market basis). Deferred to v2+. Requires understanding of credit spreads, funding conditions, and macro regimes. More complex than single-curve RV.

**Signals/Drivers:**
- SOFR-Treasury spread (OIS-Treasury basis)
- Momentum or mean-reversion on spread (TBD)

**Historical Implementation:**
- Not yet implemented
- Status: Planned (deferred to v2+)

**Candidate Modern Implementation:**
- Phase-0: Sign-only spread signals (long when SOFR rich vs Treasuries?)
- Phase-1: Z-scoring, regime-dependent logic, macro overlays
- Phase-2: Integration if passes Phase-0/1 (complex, uncertain)

**Compatibility:**
⚠️ **Sidecar / Complex Macro Strategy** — Not core Curve RV; separate research track

**Classification Justification:**
Explicitly separate from Curve RV. ROADMAP.md: "Deferred / Separate Sleeves — Cross-Market Basis RV (SOFR ↔ Treasuries): Macro RV, not pure Curve RV. Deferred to v2+."

**Extraction Confidence:** Medium

**What would invalidate this idea?**
- Phase-0 failure: Sharpe < 0.10
- SOFR-Treasury spread too stable (no trading edge)
- Credit spreads and funding conditions dominate (macro complexity too high)
- Transaction costs exceed alpha (cross-market execution difficult)

**Research questions to verify in 2026:**
- Has OIS-Treasury basis regime changed post-2023?
- Can momentum or mean-reversion on spread work?
- What macro indicators predict basis widening/narrowing?
- How does this correlate with credit spreads (HYG, LQD)?
- Can this be traded in spread format (OAS construction)?

**Needs web research?** Yes

**Web research priority:** High (OIS-Treasury basis, funding conditions, macro RV)

---

### IDEA-033: Value Meta-Sleeve (Mean-Reversion, Valuation Spreads)

**Title:** Value Meta-Sleeve (Cross-Sectional Value, Mean-Reversion)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 1150-1160)
- `docs/SOTs/STRATEGY.md` (lines 22, 183-186)

**Asset Class:** Multi-asset (Equity indices, Rates, FX, Commodities)

**Strategy Family:** Value / Mean-Reversion

**Description:**
Economic risk premia via mean-reversion signals, value spreads in commodities, deep carry interactions, bond valuation signals. Optional meta-sleeve for long-term expansion (2-3 years). Not yet designed or implemented. May include: commodity valuation (oil vs fundamentals), bond term premium, FX purchasing power parity, cross-sectional value (rich vs cheap assets).

**Signals/Drivers:**
- Mean-reversion signals (price vs moving average, z-score)
- Value spreads (commodities: oil vs natural gas, gold vs silver)
- Deep carry interactions (carry + valuation)
- Bond valuation (term premium, real yields)

**Historical Implementation:**
- Not yet implemented
- Status: Planned (optional, long-term 2-3 years)

**Candidate Modern Implementation:**
- Phase-0: Sign-only value signals (long cheap, short rich)
- Phase-1: Z-scoring, cross-sectional ranking, vol targeting
- Phase-2: Integration as standalone meta-sleeve (if passes Phase-0/1)

**Compatibility:**
✅ **Futures-Six candidate** — Optional long-term expansion, not core v1

**Classification Justification:**
Explicitly planned for long-term. ROADMAP.md: "5.2 Additional Meta-Sleeves (Optional, Long-Term) — 5.2.1 Value Meta-Sleeve" with "Economic risk premia via mean-reversion signals, value spreads in commodities, deep carry interactions, bond valuation signals."

**Extraction Confidence:** Medium

**What would invalidate this idea?**
- Phase-0 failure: Sharpe < 0.10
- Correlation with Trend exceeds 0.80 (mean-reversion vs momentum clash)
- Value signals no longer work in modern markets (efficient pricing)
- Transaction costs exceed alpha (mean-reversion requires high turnover)

**Research questions to verify in 2026:**
- Do commodity value spreads (oil vs NG, gold vs silver) contain alpha?
- Can bond term premium predict Treasury returns?
- Is FX purchasing power parity still predictive?
- How does value meta-sleeve correlate with Trend (expected negative)?
- Can value + carry interactions improve Sharpe (combo signals)?

**Needs web research?** Yes

**Web research priority:** High (value investing literature, commodity fundamentals)

---

### IDEA-034: Macro Regime Meta-Sleeve (Yield Curve, Inflation, Credit)

**Title:** Macro Regime Sleeve (Conditional Overlays)

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 1161-1167)

**Asset Class:** Overlay (applies to all strategies)

**Strategy Family:** Macro / Regime Conditioning

**Description:**
Conditional overlays based on macro indicators: yield curve steepening/flattening, inflation momentum, ISM/PPI/credit spreads filters. Used only as conditional overlays, not direct forecasts. May include: 2s10s slope regime, breakeven inflation momentum, credit spread widening (HYG vs LQD), ISM manufacturing (expansion vs contraction).

**Signals/Drivers:**
- Yield curve slope (2s10s, 10s30s)
- Inflation indicators (CPI, breakevens)
- ISM/PPI/credit spreads
- Used as regime filters, not return predictors

**Historical Implementation:**
- Partially implemented (Macro Regime Filter uses FRED indicators)
- Not yet implemented as separate meta-sleeve
- Status: Planned (optional, long-term)

**Candidate Modern Implementation:**
- Phase-0: Not applicable (overlay, not standalone strategy)
- Phase-1: Integration as regime filter for existing sleeves
- Phase-2: Validation of regime-conditioned performance

**Compatibility:**
⚠️ **Sidecar / Overlay** — Not return source; macro conditioning layer

**Classification Justification:**
Explicitly planned as overlay. ROADMAP.md: "5.2.2 Macro Regime Sleeve — Yield curve steepening/flattening, Inflation momentum, ISM/PPI/credit spreads filters. Used only as conditional overlays, not direct forecasts."

**Extraction Confidence:** Medium

**What would invalidate this idea?**
- No improvement in regime-conditioned performance (overlay not adding value)
- Macro indicators no longer predictive (regime change post-2023)
- Correlation with Macro Regime Filter exceeds 0.95 (redundancy)

**Research questions to verify in 2026:**
- Do macro indicators (ISM, credit spreads) predict strategy performance?
- Can yield curve slope improve trend sleeve timing (reduce whipsaws)?
- Is inflation momentum predictive of commodity returns?
- How does macro overlay interact with existing Macro Regime Filter?

**Needs web research?** Yes

**Web research priority:** Medium (macro indicators, regime conditioning)

---

### IDEA-035: Seasonality/Flows Meta-Sleeve

**Title:** Seasonality and Flows Meta-Sleeve

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 1186)
- `docs/SOTs/STRATEGY.md` (lines 25, 198-201)

**Asset Class:** Multi-asset (Equity indices, Rates, FX, Commodities)

**Strategy Family:** Seasonality / Flow-Driven Patterns

**Description:**
Calendar effects (month-of-year, day-of-week), flow-driven patterns (quarter-end rebalancing flows), event-driven signals. May include: commodity seasonality (heating oil winter, natural gas summer), quarter-end rebalancing (equity flows), tax-loss harvesting (December), FOMC meeting effects. Not yet designed or implemented. Long-term expansion.

**Signals/Drivers:**
- Calendar effects (month, day-of-week)
- Flow-driven patterns (quarter-end, year-end)
- Event-driven signals (FOMC, earnings)

**Historical Implementation:**
- Not yet implemented
- Status: Planned (long-term expansion, 2-3 years)

**Candidate Modern Implementation:**
- Phase-0: Sign-only seasonal signals (e.g., long heating oil in winter)
- Phase-1: Statistical validation, cross-sectional ranking
- Phase-2: Integration as standalone meta-sleeve (if passes Phase-0/1)

**Compatibility:**
✅ **Futures-Six candidate** — Optional long-term expansion, not core v1

**Classification Justification:**
Explicitly planned for long-term. ROADMAP.md: "Seasonality / Flows Meta-Sleeve — Calendar effects, flow-driven patterns, event-driven signals."

**Extraction Confidence:** Medium

**What would invalidate this idea?**
- Phase-0 failure: Sharpe < 0.10
- Seasonal patterns no longer predictive (markets adapted)
- Flow-driven patterns too noisy (transaction costs exceed alpha)
- Correlation with Trend exceeds 0.80 (redundancy)

**Research questions to verify in 2026:**
- Do commodity seasonal patterns still work (heating oil winter, NG summer)?
- Can quarter-end rebalancing flows be traded profitably?
- Is tax-loss harvesting (December) still predictive?
- How do FOMC meeting effects impact trend strategies?
- Can calendar effects improve existing sleeve timing?

**Needs web research?** Yes

**Web research priority:** Medium (seasonal patterns, flow-driven trading)

---

### IDEA-036: CSMOM Sector-Neutral Variants

**Title:** Cross-Sectional Momentum — Sector-Neutral

**Source Files:**
- `docs/SOTs/STRATEGY.md` (lines 474)
- Implied from "Future work (Phase-B enhancement)"

**Asset Class:** Multi-asset (Equity indices, Rates, FX, Commodities)

**Strategy Family:** Cross-Sectional Momentum

**Description:**
Sector-neutral CSMOM variants: rank within asset classes (commodities, rates, FX separately) instead of across all assets. May reduce single-asset concentration risk. Ensures diversification within sectors. Not yet implemented, mentioned as future Phase-B enhancement.

**Signals/Drivers:**
- Cross-sectional ranking within sectors (not across all assets)
- Sector definitions: Equities (ES, NQ, RTY), Rates (ZT, ZF, ZN, UB), FX (6E, 6B, 6J), Commodities (CL, GC)

**Historical Implementation:**
- Not yet implemented
- Status: Planned (Phase-B enhancement)

**Candidate Modern Implementation:**
- Phase-0: Sign-only sector-neutral CSMOM
- Phase-1: Vol-aware ranking, sector-specific horizon weights
- Phase-2: Integration as additional atomic sleeve within CSMOM meta

**Compatibility:**
✅ **Futures-Six candidate** — Phase-B enhancement, not core v1

**Classification Justification:**
Explicitly mentioned as future work. STRATEGY.md: "Future work (Phase-B enhancement) may introduce additional atomic sleeves for CSMOM (e.g., different lookback structures, volatility-adjusted variants, sector-neutral variants)."

**Extraction Confidence:** Medium

**What would invalidate this idea?**
- Phase-0 failure: Sharpe < 0.10
- Correlation with standard CSMOM exceeds 0.95 (no incremental benefit)
- Sector definitions arbitrary (no economic rationale)
- Within-sector ranking degenerates (too few assets per sector)

**Research questions to verify in 2026:**
- Does sector-neutral ranking improve diversification vs standard CSMOM?
- What is the optimal sector definition (asset class vs other groupings)?
- Can sector-specific horizon weights improve performance?
- How does sector-neutral CSMOM perform in small universe (13 assets)?
- Does this reduce single-asset concentration risk?

**Needs web research?** Yes

**Web research priority:** Medium (sector-neutral momentum literature)

---

### IDEA-037: Adaptive Trend Lookbacks (Regime-Dependent)

**Title:** Adaptive Trend Lookback Periods

**Source Files:**
- Implied from ROADMAP.md discussion of "Phase 4 = Engine Design Correctness"
- General principle, not explicit idea

**Asset Class:** Multi-asset (Equity indices, Rates, FX, Commodities)

**Strategy Family:** Trend / Time-Series Momentum

**Description:**
Adaptive lookback periods for trend strategies based on realized volatility or macro regime. Shorten lookback in high-vol regimes (faster response), lengthen in low-vol regimes (reduce whipsaws). May include: vol-conditioned lookback (63d in high-vol, 126d in low-vol), macro-conditioned lookback (shorten during Fed pivots), cross-sectional vol ranking (different lookbacks per asset).

**Signals/Drivers:**
- Realized volatility (21d, 63d)
- Macro regime indicators (VIX, DGS10, FEDFUNDS)
- Asset-specific vol dynamics

**Historical Implementation:**
- Not yet implemented
- Status: Conceptual (not formally proposed)

**Candidate Modern Implementation:**
- Phase-0: Test fixed vs adaptive lookback (regime-split Sharpe)
- Phase-1: Implement adaptive logic, vol-conditioned lookback rules
- Phase-2: Integration as optional enhancement (if improves Sharpe)

**Compatibility:**
✅ **Futures-Six candidate** — Phase-4+ enhancement, not core v1

**Classification Justification:**
Conceptual, implied from Phase 4 research protocol. Not explicitly documented in roadmap but consistent with "engine design correctness" philosophy.

**Extraction Confidence:** Low

**What would invalidate this idea?**
- Phase-0 failure: Adaptive lookback no better than fixed
- Overfitting to in-sample regimes (OOS performance degrades)
- Complexity not justified by marginal Sharpe improvement
- Correlation with fixed lookback exceeds 0.98 (no benefit)

**Research questions to verify in 2026:**
- Can vol-conditioned lookback improve Sharpe vs fixed?
- What is the optimal vol threshold for switching lookbacks?
- Does adaptive logic work better for short-term (21d) or long-term (252d) trend?
- How does adaptive lookback perform in 2020 and 2022 drawdowns?
- Can macro-conditioned lookback reduce trend whipsaws?

**Needs web research?** Yes

**Web research priority:** High (adaptive momentum literature, regime-dependent strategies)

---

### IDEA-038: Vol-of-Vol Aware Scaling (RT v2)

**Title:** Risk Targeting v2 — Vol-of-Vol Aware Scaling

**Source Files:**
- `docs/SOTs/ROADMAP.md` (lines 315-344)
- "Future Phase: Phase 5 — Risk Targeting v2"

**Asset Class:** Overlay (applies to all strategies)

**Strategy Family:** Risk Management / Position Sizing

**Description:**
RT v2 design to control not only portfolio volatility, but also: (1) vol-weighted asset concentration, (2) single-asset risk contribution caps, (3) tail-dominant asset scaling in crisis regimes (e.g., CL at 175% annualized vol). Origin: CSMOM_skip_v1 Phase 2 testing revealed RT v1 scales smaller positions into high-vol assets, producing 3.43x crisis risk. Not wrong, but raises question: should RT be vol-of-vol aware?

**Signals/Drivers:**
- Asset-level realized volatility (21d, 63d)
- Vol-of-vol (standard deviation of realized vol)
- Crisis regime indicators (VIX > 30, drawdown > -10%)

**Historical Implementation:**
- Not yet implemented
- Status: Planned (Phase 5, entry criteria not satisfied)

**Candidate Modern Implementation:**
- Phase-A: Define RT v2 architecture (vol-of-vol metrics, risk contribution caps)
- Phase-B: Deterministic rule-based RT v2
- Phase-C: End-to-end validation
- Phase-D: Production deployment (replace RT v1)

**Compatibility:**
⚠️ **Sidecar / RT v2 Track** — Not engine; risk targeting layer

**Classification Justification:**
Explicitly planned for Phase 5. ROADMAP.md: "Future Phase: Phase 5 — Risk Targeting v2" with "Objective: Design risk-targeting that controls not only portfolio volatility, but: (1) Vol-weighted asset concentration, (2) Single-asset risk contribution, (3) Tail-dominant asset scaling in crisis regimes."

**Extraction Confidence:** High

**What would invalidate this idea?**
- RT v2 does not improve MaxDD reduction vs RT v1
- Complexity not justified by marginal performance improvement
- Vol-of-vol metrics too noisy (false signals)
- Correlation with RT v1 exceeds 0.98 (no incremental benefit)

**Research questions to verify in 2026:**
- Can vol-of-vol metrics reduce crisis risk (2020 Q1, 2022)?
- What is the optimal vol-of-vol lookback (21d vs 63d)?
- Should RT v2 cap single-asset risk contribution (e.g., max 20%)?
- Can target_vol be belief-profile-aware (adapt to asset mix)?
- How does RT v2 interact with Allocator v1 (redundancy check)?

**Needs web research?** Yes

**Web research priority:** Medium (risk parity literature, vol-of-vol scaling)

---

## Summary Statistics

### By Status

| Status | Count |
|--------|-------|
| Active/Promoted (Production or Research-Ready) | 13 |
| Parked/Failed (Phase-0/1/2 Failures) | 14 |
| Planned/Future (Not Yet Implemented) | 11 |

### By Meta-Sleeve

| Meta-Sleeve | Active | Parked | Planned | Total |
|-------------|--------|--------|---------|-------|
| Trend | 5 | 0 | 1 | 6 |
| CSMOM | 1 | 0 | 1 | 2 |
| VRP | 2 | 7 | 3 | 12 |
| Carry | 2 | 2 | 0 | 4 |
| Curve RV | 2 | 0 | 2 | 4 |
| Crisis | 0 | 4 | 0 | 4 |
| Overlay/Other | 1 | 1 | 4 | 6 |

### Web Research Priority

| Priority | Count |
|----------|-------|
| High | 5 |
| Medium | 14 |
| Low | 3 |
| N/A | 9 |
| No | 7 |

---

## Notes on Extraction Methodology

**High Confidence Ideas:**
- Explicitly documented in SOTs with Phase-0/1/2 results
- Clear promotion or rejection status
- Source files clearly identified

**Medium Confidence Ideas:**
- Mentioned in planning documents but not yet tested
- Implied from future work sections
- Conceptual ideas without formal specification

**Low Confidence Ideas:**
- Inferred from general principles
- Not explicitly documented but consistent with system philosophy

**Excluded from Registry:**
- Internal implementation details (RT bugs, allocator thresholds)
- Data infrastructure (FRED downloads, database schemas)
- Tooling and diagnostics (waterfall analysis, checkpoint validation)

---

**End of IDEA_REGISTRY.md**
