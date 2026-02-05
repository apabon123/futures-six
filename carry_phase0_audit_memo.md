# Carry Phase-0 Inputs Audit Memo

**Date**: January 21, 2026  
**Auditor**: AI Assistant (Cursor Agent)  
**Canonical DB Path**: `C:/Users/alexp/OneDrive/Gdrive/Trading/GitHub Projects/databento-es-options/data/silver/market.duckdb`

---

## Executive Summary

**STATUS**: ❌ **HARD FAIL — Data Not Available**

Direct database inspection reveals that the canonical DuckDB contains only 3 VIX-related symbols:
- `@VX=101XN`
- `@VX=201XN`
- `@VX=301XN`

**All 30 required carry input series are missing from the database.** This is a **data loading issue**, not a symbol resolution bug.

---

## Audit Results

### Database Contents

| Metric | Value |
|--------|-------|
| **Total unique symbols in DB** | 3 |
| **Symbols found** | `@VX=101XN`, `@VX=201XN`, `@VX=301XN` |
| **Table name** | `market_data` |
| **Canonical window** | 2020-01-01 to 2025-10-31 |

### Required Keys Status

| Category | Required | Found | Missing |
|----------|----------|-------|---------|
| **Equity Spot** | 3 | 0 | 3 (SP500, NASDAQ100, RUT_SPOT) |
| **Equity Futures** | 3 | 0 | 3 (ES, NQ, RTY front contracts) |
| **Funding Rates** | 3 variants | 0 | 3 (SOFR variants) |
| **Rates Rank 1** | 4 | 0 | 4 (ZT, ZF, ZN, UB rank 1) |
| **Rates Front** | 4 | 0 | 4 (ZT, ZF, ZN, UB front) |
| **FX Front** | 3 | 0 | 3 (6E, 6B, 6J front) |
| **FX Rank 1** | 3 | 0 | 3 (6E, 6B, 6J rank 1) |
| **Commodity Front** | 2 | 0 | 2 (CL, GC front) |
| **Commodity Rank 1** | 2 | 0 | 2 (CL, GC rank 1) |
| **Foreign Rates** | 3 | 0 | 3 (ECB, JPY, SONIA) |
| **TOTAL** | **30** | **0** | **30** |

### Root Prefix Query Test

To verify if this is a symbol resolution issue vs data loading, I tested root prefix queries (as used by `get_contracts_by_root`):

- `ES%`: **0 matches**
- `NQ%`: **0 matches**
- `ZT%`: **0 matches**
- `CL%`: **0 matches**
- `6E%`: **0 matches**
- `@VX%`: **3 matches** (VIX contracts)

**Conclusion**: The database does not contain futures contract data. This is not a symbol naming/aliasing issue.

---

## Root Cause Analysis

### Is This a Symbol Resolution Bug?

**NO.** Evidence:

1. **Root prefix queries return zero results**: Even if continuous contract symbols like `ES_FRONT_CALENDAR_2D` don't exist, individual contracts matching `ES%` should exist if data were loaded.
2. **Database structure intact**: Table `market_data` exists and can be queried (VIX symbols found).
3. **No alias mapping needed**: Symbol resolution code (`get_contracts_by_root`) queries by root prefix, not exact continuous contract names.

### Is This a Data Loading Issue?

**YES.** Evidence:

1. **Only VIX data present**: Database contains 3 VIX-related symbols only.
2. **No futures root symbols**: Queries for ES%, NQ%, ZT%, CL%, etc. return zero results.
3. **Missing spot indices**: SP500, NASDAQ100, RUT_SPOT not found (these are separate data sources).
4. **Missing rate data**: SOFR, foreign rates not found (separate data sources).

---

## Required Actions (Escalation)

Per Step 3 protocol: **"If audit truly shows a key missing → escalate with exact missing keynames and DB query results; do NOT proceed to Phase-0."**

### Missing Data Inventory

**Category 1: Futures Contracts** (Individual contracts needed; continuous series built dynamically)
- Equity: ES, NQ, RTY (individual contracts like `ES H2024`, `ES M2024`, etc.)
- Rates: ZT, ZF, ZN, UB (individual contracts)
- FX: 6E, 6B, 6J (individual contracts)
- Commodity: CL, GC (individual contracts)

**Category 2: Spot Indices** (Must be loaded as separate series)
- `SP500` (S&P 500 price-return index)
- `NASDAQ100` (Nasdaq-100 price-return index)
- `RUT_SPOT` (Russell 2000 price-return index)

**Category 3: Interest Rates** (Must be loaded as separate series)
- `SOFR` (or `US_SOFR` or `SOFR_RATE`) - Secured Overnight Financing Rate
- `ECB_RATE` (or equivalent) - European Central Bank rate proxy
- `JPY_RATE` (or equivalent) - Japanese Yen rate proxy
- `SONIA` (or equivalent) - Sterling Overnight Index Average

**Category 4: Continuous Rank 1 Series** (Must be built and stored, or built on-demand)
- Rates: `ZT_RANK_1_VOLUME`, `ZF_RANK_1_VOLUME`, `ZN_RANK_1_VOLUME`, `UB_RANK_1_VOLUME`
- FX: `6E_RANK_1_CALENDAR`, `6B_RANK_1_CALENDAR`, `6J_RANK_1_CALENDAR`
- Commodity: `CL_RANK_1_VOLUME`, `GC_RANK_1_VOLUME`

**Note**: Rank 0 (front) series may be built dynamically from individual contracts, but Rank 1 series must exist or be built explicitly.

---

## Next Steps (Blocked)

❌ **Cannot proceed to Phase-0 until data is loaded.**

### Immediate Actions Required

1. **Load futures contract data** into canonical DB:
   - Individual contracts for ES, NQ, RTY, ZT, ZF, ZN, UB, 6E, 6B, 6J, CL, GC
   - Date range: At minimum 2018-01-01 to 2025-12-31 (for warmup + canonical window)

2. **Load spot indices**:
   - SP500, NASDAQ100, RUT_SPOT (price-return, NOT total return)
   - Date range: Same as futures

3. **Load interest rate data**:
   - SOFR (or equivalent)
   - Foreign rates (ECB, JPY, SONIA proxies)

4. **Build continuous rank 1 series**:
   - Build Rank 1 continuous contracts using same roll rules as Rank 0
   - Store in DB with explicit naming (e.g., `ZT_RANK_1_VOLUME`)

### After Data Loading

1. **Re-run audit**: `python scripts/diagnostics/audit_carry_inputs_coverage.py`
2. **Verify coverage**: All required keys should show ≥80% coverage over canonical window
3. **Then proceed**: Run Phase-0 diagnostic

---

## Artifacts Generated

- ✅ `carry_inputs_coverage.json` — Full audit results
- ✅ `carry_phase0_audit_memo.md` — This memo

---

## Symbol Resolution Notes (For Future Reference)

**Current Implementation**:
- `get_contracts_by_root(root="ES")` queries: `SELECT * WHERE symbol LIKE 'ES%'`
- Returns individual contracts, then assigns ranks based on symbol sorting or rank parsing
- Continuous series are built dynamically, not stored as separate symbols

**If Symbol Resolution Fixes Needed** (not applicable here):
- Would implement alias map in MarketData or feature modules
- Would log resolved DB keys during feature computation
- Would not rename DB series (per user instructions)

---

## Conclusion

The canonical DuckDB does not contain the futures contract data required for carry computation. This is a **data loading issue**, not a symbol resolution bug. All 30 required input series are missing.

**Recommendation**: Load all required futures, spot indices, and rate data before proceeding to Phase-0 testing.

---

**End of Memo**
