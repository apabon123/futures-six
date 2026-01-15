"""
Phase 3A Policy Baseline Validation

Validates that a baseline run has "teeth + determinism" per SYSTEM_CONSTRUCTION policy artifact contract.

Checks:
1. engine_policy_state_v1.csv: stress_value not-all-NaN, policy_state has OFF days
2. engine_policy_applied_v1.csv: at least one multiplier=0 (unless explicitly justified)
3. A/B weight difference proof (if baseline_run_id provided)
4. Canonical diagnostics: policy_inert=false, policy_effective=true

This is the minimal "teeth + determinism" validation for Phase 3A acceptance.
"""

import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.diagnostics.canonical_diagnostics import load_run_artifacts, generate_canonical_diagnostics


class ValidationResults:
    """Track validation results with hard failures vs soft warnings."""
    
    def __init__(self):
        self.checks = {}
        self.hard_failures = []
        self.soft_warnings = []
    
    def add_check(self, name: str, passed: bool, message: str = "", is_hard: bool = True):
        """
        Add a validation check.
        
        Args:
            name: Check name
            passed: Whether check passed
            message: Optional message
            is_hard: If True, hard failure (blocks acceptance). If False, soft warning (needs justification).
        """
        self.checks[name] = {
            'passed': passed,
            'message': message,
            'is_hard': is_hard
        }
        if not passed:
            if is_hard:
                self.hard_failures.append(name)
            else:
                self.soft_warnings.append(name)
    
    def summary(self) -> bool:
        """Print summary and return True if all hard checks passed."""
        print("\n" + "=" * 80)
        print("PHASE 3A POLICY BASELINE VALIDATION SUMMARY")
        print("=" * 80)
        
        # Hard failures first
        if self.hard_failures:
            print("\n[FAIL] HARD FAILURES (block acceptance):")
            for check_name in self.hard_failures:
                result = self.checks[check_name]
                print(f"  [FAIL] {check_name}")
                if result['message']:
                    print(f"     {result['message']}")
        
        # Soft warnings
        if self.soft_warnings:
            print("\n[WARN] SOFT WARNINGS (needs justification):")
            for check_name in self.soft_warnings:
                result = self.checks[check_name]
                print(f"  [WARN] {check_name}")
                if result['message']:
                    print(f"     {result['message']}")
        
        # Passed checks
        passed_checks = [name for name, result in self.checks.items() 
                        if result['passed']]
        if passed_checks:
            print("\n[PASS] PASSED CHECKS:")
            for check_name in passed_checks:
                result = self.checks[check_name]
                print(f"  [PASS] {check_name}")
                if result['message']:
                    print(f"     {result['message']}")
        
        print("=" * 80)
        
        if not self.hard_failures:
            if self.soft_warnings:
                print("\n[SUCCESS WITH WARNINGS] All hard checks passed, but soft warnings need justification.")
                print("Baseline run is valid for Phase 3A acceptance IF warnings are justified.")
                print("(Zero gating may be legitimate if canonical window didn't hit stress thresholds.)")
            else:
                print("\n[SUCCESS] ALL VALIDATION CHECKS PASSED")
                print("Baseline run is valid for Phase 3A acceptance.")
            return True
        else:
            print("\n[FAILURE] VALIDATION FAILED")
            print("Baseline run does not meet Phase 3A acceptance criteria.")
            return False


def validate_policy_state_artifact(run_dir: Path, results: ValidationResults):
    """Check 1: engine_policy_state_v1.csv - stress_value not-all-NaN, policy_state has OFF days."""
    print("\n--- Check 1: Policy State Artifact ---")
    
    # Check if this is a precomputed mode run (state artifact not required)
    meta_file = run_dir / "meta.json"
    is_precomputed = False
    if meta_file.exists():
        try:
            import json
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            # Precomputed mode runs have engine_policy_source_run_id
            is_precomputed = meta.get('engine_policy_source_run_id') is not None
        except Exception:
            pass
    
    state_file = run_dir / "engine_policy_state_v1.csv"
    if not state_file.exists():
        if is_precomputed:
            # Precomputed mode: state artifact not required (only applied artifact is materialized)
            results.add_check(
                "Policy State Artifact Exists",
                True,
                f"Skipped (precomputed mode - state artifact not materialized, only applied artifact required)"
            )
            return
        else:
            # Compute mode: state artifact is required
            results.add_check(
                "Policy State Artifact Exists",
                False,
                f"engine_policy_state_v1.csv not found in {run_dir} (required for compute mode)"
            )
            return
    
    try:
        state_df = pd.read_csv(state_file, parse_dates=['date'])
        
        # Check required columns
        required_cols = ['date', 'engine', 'stress_value', 'policy_state', 'policy_multiplier']
        missing_cols = [col for col in required_cols if col not in state_df.columns]
        if missing_cols:
            results.add_check(
                "Policy State Schema",
                False,
                f"Missing columns: {', '.join(missing_cols)}"
            )
            return
        
        # Check for Trend and VRP engines
        engines = state_df['engine'].unique()
        has_trend = 'trend' in engines
        has_vrp = 'vrp' in engines
        
        if not has_trend and not has_vrp:
            results.add_check(
                "Policy State Engines",
                False,
                "No 'trend' or 'vrp' engines found in state artifact"
            )
            return
        
        # Check stress_value not-all-NaN for each engine
        for engine_name in ['trend', 'vrp']:
            if engine_name not in engines:
                continue
            
            engine_state = state_df[state_df['engine'] == engine_name]
            stress_values = engine_state['stress_value']
            
            # Check not-all-NaN
            n_valid = stress_values.notna().sum()
            n_total = len(stress_values)
            
            if n_valid == 0:
                results.add_check(
                    f"Policy State: {engine_name} stress_value has data",
                    False,
                    f"All {n_total} stress_value entries are NaN for {engine_name} engine",
                    is_hard=True  # Hard failure: features are all-NaN
                )
            else:
                results.add_check(
                    f"Policy State: {engine_name} stress_value has data",
                    True,
                    f"{n_valid}/{n_total} stress_value entries are valid for {engine_name} engine"
                )
            
            # Check policy_state has OFF days
            # SOFT WARNING: Zero gating may be legitimate if canonical window didn't hit stress thresholds
            policy_states = engine_state['policy_state']
            has_off = (policy_states == 'OFF').any()
            
            if not has_off:
                results.add_check(
                    f"Policy State: {engine_name} has OFF days",
                    False,
                    f"No OFF days found for {engine_name} engine (policy never gated). "
                    f"May be legitimate if canonical window didn't hit stress thresholds - needs justification.",
                    is_hard=False  # Soft warning, not hard failure
                )
            else:
                n_off = (policy_states == 'OFF').sum()
                results.add_check(
                    f"Policy State: {engine_name} has OFF days",
                    True,
                    f"{n_off} OFF days found for {engine_name} engine"
                )
        
    except Exception as e:
        results.add_check(
            "Policy State Artifact Read",
            False,
            f"Error reading engine_policy_state_v1.csv: {e}"
        )


def validate_policy_applied_artifact(run_dir: Path, results: ValidationResults):
    """Check 2: engine_policy_applied_v1.csv - at least one multiplier=0."""
    print("\n--- Check 2: Policy Applied Artifact ---")
    
    applied_file = run_dir / "engine_policy_applied_v1.csv"
    if not applied_file.exists():
        results.add_check(
            "Policy Applied Artifact Exists",
            False,
            f"engine_policy_applied_v1.csv not found in {run_dir}"
        )
        return
    
    try:
        applied_df = pd.read_csv(applied_file, parse_dates=['rebalance_date'])
        
        # Check required columns (canonical pivot format)
        required_cols = ['rebalance_date']
        missing_cols = [col for col in required_cols if col not in applied_df.columns]
        if missing_cols:
            results.add_check(
                "Policy Applied Schema",
                False,
                f"Missing columns: {', '.join(missing_cols)}"
            )
            return
        
        # Check for multiplier columns
        has_trend_mult = 'trend_multiplier' in applied_df.columns
        has_vrp_mult = 'vrp_multiplier' in applied_df.columns
        
        if not has_trend_mult and not has_vrp_mult:
            results.add_check(
                "Policy Applied Multipliers",
                False,
                "No trend_multiplier or vrp_multiplier columns found"
            )
            return
        
        # Check at least one multiplier=0 for each engine
        for engine_name, mult_col in [('trend', 'trend_multiplier'), ('vrp', 'vrp_multiplier')]:
            if mult_col not in applied_df.columns:
                continue
            
            multipliers = applied_df[mult_col]
            gated = (multipliers < 0.999).sum()  # Slightly below 1.0 to account for rounding
            total = len(multipliers)
            
            if gated == 0:
                results.add_check(
                    f"Policy Applied: {engine_name} has gating",
                    False,
                    f"No gating found for {engine_name} (all {total} multipliers = 1.0). "
                    f"May be legitimate if canonical window didn't hit stress thresholds - needs justification.",
                    is_hard=False  # Soft warning, not hard failure
                )
            else:
                pct = (gated / total * 100) if total > 0 else 0.0
                results.add_check(
                    f"Policy Applied: {engine_name} has gating",
                    True,
                    f"{gated}/{total} rebalances gated ({pct:.1f}%) for {engine_name}"
                )
        
    except Exception as e:
        results.add_check(
            "Policy Applied Artifact Read",
            False,
            f"Error reading engine_policy_applied_v1.csv: {e}"
        )


def validate_weight_difference_proof(run_dir: Path, baseline_run_id: str, results: ValidationResults):
    """Check 3: A/B weight difference proof - weights differ when stress triggers."""
    print("\n--- Check 3: A/B Weight Difference Proof ---")
    
    if not baseline_run_id:
        results.add_check(
            "Weight Difference Proof",
            True,
            "Skipped (no baseline_run_id provided)"
        )
        return
    
    runs_dir = run_dir.parent
    baseline_run_dir = runs_dir / baseline_run_id
    
    if not baseline_run_dir.exists():
        results.add_check(
            "Weight Difference Proof: Baseline Exists",
            False,
            f"Baseline run directory not found: {baseline_run_dir}"
        )
        return
    
    # Load weights from both runs
    current_weights_file = run_dir / "weights.csv"
    baseline_weights_file = baseline_run_dir / "weights.csv"
    
    if not current_weights_file.exists():
        results.add_check(
            "Weight Difference Proof: Current Weights",
            False,
            f"weights.csv not found in {run_dir}"
        )
        return
    
    if not baseline_weights_file.exists():
        results.add_check(
            "Weight Difference Proof: Baseline Weights",
            False,
            f"weights.csv not found in {baseline_run_dir}"
        )
        return
    
    try:
        current_weights = pd.read_csv(current_weights_file, index_col=0, parse_dates=True)
        baseline_weights = pd.read_csv(baseline_weights_file, index_col=0, parse_dates=True)
        
        # Load policy applied to find gated dates
        applied_file = run_dir / "engine_policy_applied_v1.csv"
        if not applied_file.exists():
            results.add_check(
                "Weight Difference Proof: Policy Applied",
                False,
                "Cannot check weight differences without engine_policy_applied_v1.csv"
            )
            return
        
        applied_df = pd.read_csv(applied_file, parse_dates=['rebalance_date'])
        applied_df = applied_df.set_index('rebalance_date')
        
        # Find dates where policy gated (multiplier < 1.0)
        gated_dates = set()
        if 'trend_multiplier' in applied_df.columns:
            trend_gated = applied_df[applied_df['trend_multiplier'] < 0.999].index
            gated_dates.update(trend_gated)
        if 'vrp_multiplier' in applied_df.columns:
            vrp_gated = applied_df[applied_df['vrp_multiplier'] < 0.999].index
            gated_dates.update(vrp_gated)
        
        if not gated_dates:
            results.add_check(
                "Weight Difference Proof: Gated Dates",
                False,
                "No gated dates found (policy never triggered)"
            )
            return
        
        # Align weights to common dates
        common_dates = set(current_weights.index) & set(baseline_weights.index) & gated_dates
        
        if not common_dates:
            results.add_check(
                "Weight Difference Proof: Common Dates",
                False,
                "No common dates found between runs on gated dates"
            )
            return
        
        # Check if weights differ on at least one gated date
        weights_differ = False
        for date in sorted(common_dates)[:10]:  # Check first 10 gated dates
            current_w = current_weights.loc[date]
            baseline_w = baseline_weights.loc[date]
            
            # Check if weights differ (allowing for small numerical differences)
            if not current_w.equals(baseline_w):
                diff = (current_w - baseline_w).abs().max()
                if diff > 1e-6:  # Significant difference
                    weights_differ = True
                    break
        
        if weights_differ:
            results.add_check(
                "Weight Difference Proof: Weights Differ on Gated Dates",
                True,
                f"Weights differ on at least one gated date (checked {min(10, len(common_dates))} dates)"
            )
        else:
            results.add_check(
                "Weight Difference Proof: Weights Differ on Gated Dates",
                False,
                f"Weights do not differ on any of the {len(common_dates)} gated dates checked"
            )
        
    except Exception as e:
        results.add_check(
            "Weight Difference Proof: Execution",
            False,
            f"Error comparing weights: {e}"
        )


def validate_canonical_diagnostics(run_dir: Path, results: ValidationResults):
    """Check 4: Canonical diagnostics - policy_inert=false, policy_effective=true."""
    print("\n--- Check 4: Canonical Diagnostics Governance ---")
    
    # Generate diagnostics if not present
    diagnostics_file = run_dir / "canonical_diagnostics.json"
    if not diagnostics_file.exists():
        # Try to generate
        run_id = run_dir.name
        try:
            report = generate_canonical_diagnostics(run_id, run_dir)
            # Save if generated
            with open(diagnostics_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            results.add_check(
                "Canonical Diagnostics: Generation",
                False,
                f"Could not generate canonical diagnostics: {e}"
            )
            return
    
    try:
        with open(diagnostics_file, 'r') as f:
            diagnostics = json.load(f)
        
        binding = diagnostics.get('constraint_binding', {})
        
        # Check policy_inert
        # HARD FAILURE: policy_inert=true blocks acceptance
        policy_inert = binding.get('policy_inert', None)
        if policy_inert is None:
            results.add_check(
                "Canonical Diagnostics: policy_inert field",
                False,
                "policy_inert field missing from constraint_binding",
                is_hard=True
            )
        elif policy_inert:
            policy_inert_reason = binding.get('policy_inert_reason', 'Unknown')
            results.add_check(
                "Canonical Diagnostics: policy_inert=false",
                False,
                f"Run is Policy-Inert: {policy_inert_reason}",
                is_hard=True  # Hard failure: policy is inert
            )
        else:
            results.add_check(
                "Canonical Diagnostics: policy_inert=false",
                True,
                "Run is not Policy-Inert"
            )
        
        # Check policy_effective
        # HARD FAILURE: policy_effective=false blocks acceptance
        policy_effective = binding.get('policy_effective', None)
        if policy_effective is None:
            results.add_check(
                "Canonical Diagnostics: policy_effective field",
                False,
                "policy_effective field missing from constraint_binding",
                is_hard=True
            )
        elif not policy_effective:
            results.add_check(
                "Canonical Diagnostics: policy_effective=true",
                False,
                "Policy is not effective (enabled but inputs missing or no teeth)",
                is_hard=True  # Hard failure: policy not effective
            )
        else:
            results.add_check(
                "Canonical Diagnostics: policy_effective=true",
                True,
                "Policy is effective"
            )
        
    except Exception as e:
        results.add_check(
            "Canonical Diagnostics: Read",
            False,
            f"Error reading canonical_diagnostics.json: {e}"
        )


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Phase 3A Policy Baseline (teeth + determinism)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'run_id',
        type=str,
        help='Run ID to validate'
    )
    
    parser.add_argument(
        '--baseline_run_id',
        type=str,
        default=None,
        help='Baseline run ID for A/B weight difference proof (optional)'
    )
    
    parser.add_argument(
        '--runs_dir',
        type=str,
        default='reports/runs',
        help='Base directory for runs (default: reports/runs)'
    )
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    run_dir = runs_dir / args.run_id
    
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"PHASE 3A POLICY BASELINE VALIDATION")
    print("=" * 80)
    print(f"Run ID: {args.run_id}")
    print(f"Run Directory: {run_dir}")
    if args.baseline_run_id:
        print(f"Baseline Run ID: {args.baseline_run_id}")
    print("=" * 80)
    
    results = ValidationResults()
    
    # Run all checks
    validate_policy_state_artifact(run_dir, results)
    validate_policy_applied_artifact(run_dir, results)
    validate_weight_difference_proof(run_dir, args.baseline_run_id, results)
    validate_canonical_diagnostics(run_dir, results)
    
    # Summary
    all_passed = results.summary()
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
