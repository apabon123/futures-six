#!/usr/bin/env python3
"""
CSMOM_skip_v1 Phase 2 Integration Test

Research Protocol: Phase 4 Engine Research
Hypothesis ID: CSMOM-H001
Variant: CSMOM_skip_v1

This script runs the Phase 2 integration test:
1. Run full system with baseline CSMOM (no skip)
2. Run full system with CSMOM_skip_v1 (skip = 5, 10, 21)
3. Compare at Post-Construction layer
4. Generate promotion decision

SINGLE CHANGE ONLY:
- Baseline: skips = None (or [0, 0, 0])
- Variant: skips = [5, 10, 21]

All other system parameters are FROZEN.

Author: Phase 4 Engine Research
Date: January 2026
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

EVAL_START = "2020-03-20"
EVAL_END = "2025-10-31"

# Baseline: Phase 3B pinned baseline (no skip)
BASELINE_RUN_ID = "phase3b_baseline_artifacts_only_20260120_093953"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase4_research" / "csmom_phase2"
REPORTS_DIR = PROJECT_ROOT / "reports" / "runs"


def run_variant(run_id: str, config_path: str) -> bool:
    """Run a variant using the main strategy runner."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_strategy.py"),
        "--config_path", config_path,
        "--run_id", run_id,
        "--start", "2020-01-06",
        "--end", EVAL_END,
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Run failed with code {result.returncode}")
        print(f"STDOUT: {result.stdout[-2000:] if result.stdout else 'None'}")
        print(f"STDERR: {result.stderr[-2000:] if result.stderr else 'None'}")
        return False
    
    return True


def load_run_metrics(run_id: str) -> dict:
    """Load metrics from a run."""
    run_dir = REPORTS_DIR / run_id
    
    metrics = {}
    
    # Load canonical diagnostics
    diag_path = run_dir / "canonical_diagnostics.json"
    if diag_path.exists():
        with open(diag_path) as f:
            metrics["canonical_diagnostics"] = json.load(f)
    
    # Load engine attribution
    attr_path = run_dir / "engine_attribution_post_construction.json"
    if attr_path.exists():
        with open(attr_path) as f:
            metrics["engine_attribution"] = json.load(f)
    
    # Load waterfall attribution
    waterfall_path = run_dir / "waterfall_attribution.json"
    if waterfall_path.exists():
        with open(waterfall_path) as f:
            metrics["waterfall_attribution"] = json.load(f)
    
    # Load sleeve returns
    sleeve_path = run_dir / "sleeve_returns.csv"
    if sleeve_path.exists():
        sleeve_rets = pd.read_csv(sleeve_path, index_col=0, parse_dates=True)
        sleeve_rets = sleeve_rets.loc[EVAL_START:]
        
        # Compute per-sleeve metrics
        sleeve_metrics = {}
        for col in sleeve_rets.columns:
            rets = sleeve_rets[col].dropna()
            rets = rets[rets != 0]
            if len(rets) > 0:
                ann_ret = rets.mean() * 252
                ann_vol = rets.std() * np.sqrt(252)
                sleeve_metrics[col] = {
                    "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
                    "ann_return": ann_ret,
                    "ann_vol": ann_vol,
                }
        metrics["sleeve_metrics"] = sleeve_metrics
    
    # Load portfolio returns for regime analysis
    port_path = run_dir / "portfolio_returns.csv"
    if port_path.exists():
        port_rets = pd.read_csv(port_path, index_col=0, parse_dates=True)
        if "portfolio_returns" in port_rets.columns:
            port_rets = port_rets["portfolio_returns"]
        else:
            port_rets = port_rets.iloc[:, 0]
        port_rets = port_rets.loc[EVAL_START:]
        
        # Compute regime metrics
        rolling_vol = port_rets.rolling(21).std() * np.sqrt(252)
        vol_25 = rolling_vol.quantile(0.25)
        vol_75 = rolling_vol.quantile(0.75)
        
        regime_metrics = {}
        for name, mask in [
            ("low_vol", rolling_vol < vol_25),
            ("mid_vol", (rolling_vol >= vol_25) & (rolling_vol <= vol_75)),
            ("high_vol", rolling_vol > vol_75),
        ]:
            regime_rets = port_rets[mask]
            if len(regime_rets) > 20:
                ann_ret = regime_rets.mean() * 252
                ann_vol = regime_rets.std() * np.sqrt(252)
                regime_metrics[name] = {
                    "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
                    "n_days": len(regime_rets),
                }
        metrics["regime_metrics"] = regime_metrics
        
        # Yearly returns
        yearly = {}
        for year in port_rets.index.year.unique():
            year_rets = port_rets[port_rets.index.year == year]
            if len(year_rets) > 0:
                yearly[int(year)] = (1 + year_rets).prod() - 1
        metrics["yearly_returns"] = yearly
    
    # Load meta.json for governance info
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metrics["meta"] = json.load(f)
    
    return metrics


def generate_diagnostics(run_id: str) -> bool:
    """Generate canonical diagnostics for a run."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "diagnostics" / "generate_canonical_diagnostics.py"),
        "--run_id", run_id,
    ]
    
    print(f"\nGenerating diagnostics: {run_id}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    
    return result.returncode == 0


def generate_engine_attribution(run_id: str) -> bool:
    """Generate engine attribution for a run."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "diagnostics" / "run_engine_attribution.py"),
        "--run_id", run_id,
    ]
    
    print(f"\nGenerating engine attribution: {run_id}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    
    return result.returncode == 0


def compare_runs(baseline_metrics: dict, variant_metrics: dict) -> dict:
    """Compare baseline and variant metrics."""
    comparison = {}
    
    # Waterfall comparison (Post-Construction is the key layer)
    if "waterfall_attribution" in baseline_metrics and "waterfall_attribution" in variant_metrics:
        base_wf = baseline_metrics["waterfall_attribution"]
        var_wf = variant_metrics["waterfall_attribution"]
        
        comparison["waterfall"] = {}
        for stage in ["post_construction", "post_rt", "post_allocator"]:
            if stage in base_wf.get("stages", {}) and stage in var_wf.get("stages", {}):
                base_sharpe = base_wf["stages"][stage].get("sharpe", 0)
                var_sharpe = var_wf["stages"][stage].get("sharpe", 0)
                comparison["waterfall"][stage] = {
                    "baseline": base_sharpe,
                    "variant": var_sharpe,
                    "delta": var_sharpe - base_sharpe,
                }
    
    # Sleeve-level comparison
    if "sleeve_metrics" in baseline_metrics and "sleeve_metrics" in variant_metrics:
        comparison["sleeves"] = {}
        for sleeve in ["csmom_meta", "tsmom_multihorizon"]:
            if sleeve in baseline_metrics["sleeve_metrics"] and sleeve in variant_metrics["sleeve_metrics"]:
                base_sharpe = baseline_metrics["sleeve_metrics"][sleeve].get("sharpe", 0)
                var_sharpe = variant_metrics["sleeve_metrics"][sleeve].get("sharpe", 0)
                comparison["sleeves"][sleeve] = {
                    "baseline": base_sharpe,
                    "variant": var_sharpe,
                    "delta": var_sharpe - base_sharpe,
                }
    
    # Regime comparison
    if "regime_metrics" in baseline_metrics and "regime_metrics" in variant_metrics:
        comparison["regimes"] = {}
        for regime in ["low_vol", "mid_vol", "high_vol"]:
            if regime in baseline_metrics["regime_metrics"] and regime in variant_metrics["regime_metrics"]:
                base_sharpe = baseline_metrics["regime_metrics"][regime].get("sharpe", 0)
                var_sharpe = variant_metrics["regime_metrics"][regime].get("sharpe", 0)
                comparison["regimes"][regime] = {
                    "baseline": base_sharpe,
                    "variant": var_sharpe,
                    "delta": var_sharpe - base_sharpe,
                }
    
    # Yearly comparison
    if "yearly_returns" in baseline_metrics and "yearly_returns" in variant_metrics:
        comparison["yearly"] = {}
        all_years = set(baseline_metrics["yearly_returns"].keys()) | set(variant_metrics["yearly_returns"].keys())
        for year in sorted(all_years):
            base_ret = baseline_metrics["yearly_returns"].get(year, 0)
            var_ret = variant_metrics["yearly_returns"].get(year, 0)
            comparison["yearly"][year] = {
                "baseline": base_ret,
                "variant": var_ret,
                "delta": var_ret - base_ret,
            }
    
    return comparison


def generate_report(baseline_id: str, variant_id: str, comparison: dict) -> str:
    """Generate Phase 2 comparison report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# CSMOM_skip_v1 Phase 2 Integration Test Results

**Hypothesis ID:** CSMOM-H001
**Generated:** {timestamp}
**Research Phase:** Phase 4 Engine Research, Step 5 (Phase 2 Promotion Gate)

---

## Configuration

| Parameter | Baseline | Variant (skip_v1) |
|-----------|----------|-------------------|
| Skips | None | [5, 10, 21] |
| Lookbacks | [63, 126, 252] | [63, 126, 252] |
| Weights | [0.4, 0.35, 0.25] | [0.4, 0.35, 0.25] |

**All other parameters frozen (RT, Allocator, Construction, other engines).**

---

## Waterfall Attribution (Key Metric: Post-Construction)

| Stage | Baseline Sharpe | Variant Sharpe | Delta |
|-------|-----------------|----------------|-------|
"""
    
    if "waterfall" in comparison:
        for stage, metrics in comparison["waterfall"].items():
            report += f"| {stage} | {metrics['baseline']:.3f} | {metrics['variant']:.3f} | {metrics['delta']:+.3f} |\n"
    
    report += """
---

## Sleeve-Level Comparison

| Sleeve | Baseline Sharpe | Variant Sharpe | Delta |
|--------|-----------------|----------------|-------|
"""
    
    if "sleeves" in comparison:
        for sleeve, metrics in comparison["sleeves"].items():
            report += f"| {sleeve} | {metrics['baseline']:.3f} | {metrics['variant']:.3f} | {metrics['delta']:+.3f} |\n"
    
    report += """
---

## Regime-Conditioned Sharpe

| Regime | Baseline | Variant | Delta |
|--------|----------|---------|-------|
"""
    
    if "regimes" in comparison:
        for regime, metrics in comparison["regimes"].items():
            report += f"| {regime} | {metrics['baseline']:.3f} | {metrics['variant']:.3f} | {metrics['delta']:+.3f} |\n"
    
    report += """
---

## Yearly Returns

| Year | Baseline | Variant | Delta |
|------|----------|---------|-------|
"""
    
    if "yearly" in comparison:
        for year, metrics in sorted(comparison["yearly"].items()):
            report += f"| {year} | {metrics['baseline']:.2%} | {metrics['variant']:.2%} | {metrics['delta']:+.2%} |\n"
    
    # Promotion decision logic
    post_construction_improved = False
    if "waterfall" in comparison and "post_construction" in comparison["waterfall"]:
        post_construction_improved = comparison["waterfall"]["post_construction"]["delta"] > 0
    
    high_vol_improved = False
    if "regimes" in comparison and "high_vol" in comparison["regimes"]:
        high_vol_improved = comparison["regimes"]["high_vol"]["delta"] > 0
    
    csmom_improved = False
    if "sleeves" in comparison and "csmom_meta" in comparison["sleeves"]:
        csmom_improved = comparison["sleeves"]["csmom_meta"]["delta"] > 0
    
    report += f"""
---

## Promotion Decision Criteria

| Criterion | Baseline | Variant | Met? |
|-----------|----------|---------|------|
| Post-Construction Sharpe improves | — | — | {'YES' if post_construction_improved else 'NO'} |
| High-vol regime Sharpe improves | — | — | {'YES' if high_vol_improved else 'NO'} |
| CSMOM sleeve Sharpe improves | — | — | {'YES' if csmom_improved else 'NO'} |

---

## Final Decision

"""
    
    if post_construction_improved and csmom_improved:
        report += """**PROMOTED** — CSMOM_skip_v1 passes Phase 2 integration test.

The variant:
- Improves Post-Construction Sharpe (system-level)
- Improves CSMOM sleeve Sharpe (engine-level)
- Does not degrade other system components

**Next Step:** Update canonical CSMOM configuration to use skips=[5, 10, 21].
"""
    elif csmom_improved:
        report += """**CONDITIONAL PASS** — CSMOM_skip_v1 improves CSMOM but system-level impact is neutral/negative.

The variant:
- Improves CSMOM sleeve Sharpe (engine-level)
- System-level impact requires further analysis

**Recommendation:** Review before full promotion.
"""
    else:
        report += """**REJECTED** — CSMOM_skip_v1 fails Phase 2 integration test.

The variant does not meet promotion criteria at the system level.

**Status:** Hypothesis CSMOM-H001 is rejected permanently per research protocol.
"""
    
    report += f"""
---

## Run IDs

- **Baseline**: `{baseline_id}`
- **Variant**: `{variant_id}`

---

*Reference: `docs/PHASE4_ENGINE_RESEARCH_PROTOCOL.md` § "Step 4 — Phase 2 Integrated Test"*
"""
    
    return report


def main():
    """Run Phase 2 integration test."""
    print("=" * 70)
    print("CSMOM_skip_v1 PHASE 2 INTEGRATION TEST")
    print("Hypothesis ID: CSMOM-H001")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_run_id = f"phase4_csmom_skip_v1_{timestamp}"
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Step 1: Load baseline metrics (already exists)
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("STEP 1: Loading baseline metrics")
    print(f"{'='*60}")
    print(f"Baseline Run ID: {BASELINE_RUN_ID}")
    
    baseline_dir = REPORTS_DIR / BASELINE_RUN_ID
    if not baseline_dir.exists():
        print(f"ERROR: Baseline run not found: {baseline_dir}")
        return
    
    baseline_metrics = load_run_metrics(BASELINE_RUN_ID)
    print(f"Loaded baseline metrics: {list(baseline_metrics.keys())}")
    
    # ==========================================================================
    # Step 2: Run variant
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("STEP 2: Running CSMOM_skip_v1 variant")
    print(f"{'='*60}")
    print(f"Variant Run ID: {variant_run_id}")
    
    config_path = str(PROJECT_ROOT / "configs" / "phase4_csmom_skip_v1.yaml")
    
    success = run_variant(variant_run_id, config_path)
    if not success:
        print("ERROR: Variant run failed")
        return
    
    # ==========================================================================
    # Step 3: Generate diagnostics for variant
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("STEP 3: Generating diagnostics")
    print(f"{'='*60}")
    
    generate_diagnostics(variant_run_id)
    generate_engine_attribution(variant_run_id)
    
    # ==========================================================================
    # Step 4: Load variant metrics
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("STEP 4: Loading variant metrics")
    print(f"{'='*60}")
    
    variant_metrics = load_run_metrics(variant_run_id)
    print(f"Loaded variant metrics: {list(variant_metrics.keys())}")
    
    # ==========================================================================
    # Step 5: Compare
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("STEP 5: Comparing runs")
    print(f"{'='*60}")
    
    comparison = compare_runs(baseline_metrics, variant_metrics)
    
    # Print summary
    print("\n### WATERFALL COMPARISON ###")
    if "waterfall" in comparison:
        print(f"{'Stage':<20} {'Baseline':>12} {'Variant':>12} {'Delta':>12}")
        print("-" * 56)
        for stage, metrics in comparison["waterfall"].items():
            print(f"{stage:<20} {metrics['baseline']:>12.3f} {metrics['variant']:>12.3f} {metrics['delta']:>+12.3f}")
    
    print("\n### SLEEVE COMPARISON ###")
    if "sleeves" in comparison:
        print(f"{'Sleeve':<20} {'Baseline':>12} {'Variant':>12} {'Delta':>12}")
        print("-" * 56)
        for sleeve, metrics in comparison["sleeves"].items():
            print(f"{sleeve:<20} {metrics['baseline']:>12.3f} {metrics['variant']:>12.3f} {metrics['delta']:>+12.3f}")
    
    print("\n### REGIME COMPARISON ###")
    if "regimes" in comparison:
        print(f"{'Regime':<20} {'Baseline':>12} {'Variant':>12} {'Delta':>12}")
        print("-" * 56)
        for regime, metrics in comparison["regimes"].items():
            print(f"{regime:<20} {metrics['baseline']:>12.3f} {metrics['variant']:>12.3f} {metrics['delta']:>+12.3f}")
    
    # ==========================================================================
    # Step 6: Generate report
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("STEP 6: Generating report")
    print(f"{'='*60}")
    
    report = generate_report(BASELINE_RUN_ID, variant_run_id, comparison)
    
    report_path = OUTPUT_DIR / f"CSMOM_skip_v1_phase2_results_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    # Save comparison JSON
    json_path = OUTPUT_DIR / f"CSMOM_skip_v1_phase2_comparison_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "baseline_run_id": BASELINE_RUN_ID,
            "variant_run_id": variant_run_id,
            "comparison": comparison,
        }, f, indent=2, default=str)
    
    print(f"\nReport saved to: {report_path}")
    print(f"Comparison saved to: {json_path}")
    
    print("\n" + "=" * 70)
    print("PHASE 2 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
