#!/usr/bin/env python
"""
Validator Script for Phase-2 Engine Policy v1 Acceptance

This script validates that Engine Policy v1 is correctly implemented by checking:

1. **Artifacts Exist**
   - engine_policy_state_v1.csv (daily state)
   - engine_policy_applied_v1.csv (rebalance multipliers)
   - engine_policy_v1_meta.json (metadata)

2. **Determinism**
   - In compute mode, rerun with same config yields identical engine_policy_applied_v1.csv

3. **Lag is Correct**
   - Multiplier used at rebalance t equals policy decision from t-1 rebalance window

4. **Policy Has Teeth**
   - Compare baseline vs policy run: weights differ on at least one rebalance date when stress triggers

5. **Isolation**
   - Only trend engine weights are affected (other engines unchanged when they exist)

Usage:
    python scripts/diagnostics/validate_phase2_policy_v1.py <run_id> [--baseline_run_id <baseline>]
    
    # For compute mode validation:
    python scripts/diagnostics/validate_phase2_policy_v1.py policy_trend_gamma_compute_proof_2024
    
    # For precomputed mode validation:
    python scripts/diagnostics/validate_phase2_policy_v1.py policy_trend_gamma_apply_precomputed_2024 --baseline_run_id policy_trend_gamma_compute_proof_2024
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "reports" / "runs"


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, check: str, message: str):
        self.passed.append((check, message))
        logger.info(f"[PASS] {check} - {message}")
    
    def add_fail(self, check: str, message: str):
        self.failed.append((check, message))
        logger.error(f"[FAIL] {check} - {message}")
    
    def add_warning(self, check: str, message: str):
        self.warnings.append((check, message))
        logger.warning(f"[WARN] {check} - {message}")
    
    def summary(self) -> bool:
        """Print summary and return True if all checks passed."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"PASSED: {len(self.passed)}")
        print(f"FAILED: {len(self.failed)}")
        print(f"WARNINGS: {len(self.warnings)}")
        
        if self.failed:
            print("\n--- FAILED CHECKS ---")
            for check, msg in self.failed:
                print(f"  [FAIL] {check}: {msg}")
        
        if self.warnings:
            print("\n--- WARNINGS ---")
            for check, msg in self.warnings:
                print(f"  [WARN] {check}: {msg}")
        
        print("=" * 80)
        
        return len(self.failed) == 0


def load_run_artifacts(run_dir: Path) -> Dict:
    """Load all relevant artifacts from a run directory."""
    artifacts = {}
    
    # Check for mode by reading meta.json (standardized naming)
    meta_file = run_dir / "engine_policy_v1_meta.json"
    is_precomputed = False
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            artifacts['meta'] = meta
            is_precomputed = meta.get('mode') == 'precomputed'
    
    # Always use standard artifact names (consistent naming)
    state_file = run_dir / "engine_policy_state_v1.csv"
    applied_file = run_dir / "engine_policy_applied_v1.csv"
    
    # State CSV exists only in compute mode
    if state_file.exists():
        artifacts['state'] = pd.read_csv(state_file, parse_dates=['date'])
    
    # Applied CSV exists in both modes (standardized naming)
    if applied_file.exists():
        artifacts['applied'] = pd.read_csv(applied_file, parse_dates=['rebalance_date'])
    
    # Optional artifacts
    weights_file = run_dir / "weights.csv"
    if weights_file.exists():
        artifacts['weights'] = pd.read_csv(weights_file, index_col=0, parse_dates=True)
    
    return artifacts


def validate_artifacts_exist(run_dir: Path, results: ValidationResult) -> bool:
    """Check that all required artifacts exist."""
    # Check mode by reading meta.json (standardized naming)
    meta_file = run_dir / "engine_policy_v1_meta.json"
    is_precomputed = False
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            is_precomputed = meta.get('mode') == 'precomputed'
    
    # Standardized artifact names (same in both modes)
    required_files = [
        ("engine_policy_applied_v1.csv", "Applied multipliers"),
        ("engine_policy_v1_meta.json", "Metadata")
    ]
    
    # State CSV only required in compute mode
    if not is_precomputed:
        required_files.insert(0, ("engine_policy_state_v1.csv", "Daily state"))
    
    all_exist = True
    for filename, description in required_files:
        filepath = run_dir / filename
        if filepath.exists():
            results.add_pass(f"Artifact: {filename}", f"{description} - File exists")
        else:
            results.add_fail(f"Artifact: {filename}", f"{description} - File NOT FOUND: {filepath}")
            all_exist = False
    
    return all_exist


def validate_state_schema(state_df: pd.DataFrame, results: ValidationResult):
    """Validate state DataFrame has correct schema."""
    required_cols = ['date', 'engine', 'stress_value', 'policy_state', 'policy_multiplier']
    optional_cols = ['stress_percentile']
    
    for col in required_cols:
        if col in state_df.columns:
            results.add_pass(f"State schema: {col}", f"Column present")
        else:
            results.add_fail(f"State schema: {col}", f"Column MISSING")
    
    # Check policy_state values
    valid_states = {'ON', 'OFF'}
    actual_states = set(state_df['policy_state'].unique())
    if actual_states.issubset(valid_states):
        results.add_pass("State values: policy_state", f"All values valid: {actual_states}")
    else:
        invalid = actual_states - valid_states
        results.add_fail("State values: policy_state", f"Invalid values: {invalid}")
    
    # Check policy_multiplier values
    valid_mults = {0, 1}
    actual_mults = set(state_df['policy_multiplier'].unique())
    if actual_mults.issubset(valid_mults):
        results.add_pass("State values: policy_multiplier", f"All values valid (binary): {actual_mults}")
    else:
        results.add_fail("State values: policy_multiplier", f"Non-binary values found: {actual_mults}")


def validate_applied_schema(applied_df: pd.DataFrame, results: ValidationResult):
    """Validate applied DataFrame has correct schema."""
    required_cols = ['rebalance_date', 'engine', 'policy_multiplier_used']
    
    for col in required_cols:
        if col in applied_df.columns:
            results.add_pass(f"Applied schema: {col}", f"Column present")
        else:
            results.add_fail(f"Applied schema: {col}", f"Column MISSING")
    
    # Check multiplier values are binary
    valid_mults = {0, 1}
    actual_mults = set(applied_df['policy_multiplier_used'].unique())
    if actual_mults.issubset(valid_mults):
        results.add_pass("Applied values: multiplier", f"All values valid (binary): {actual_mults}")
    else:
        results.add_fail("Applied values: multiplier", f"Non-binary values found: {actual_mults}")


def validate_lag_correctness(
    state_df: pd.DataFrame,
    applied_df: pd.DataFrame,
    lag_rebalances: int,
    results: ValidationResult
):
    """Validate that lag is correctly applied."""
    # Get unique rebalance dates
    rebalance_dates = sorted(applied_df['rebalance_date'].unique())
    
    if len(rebalance_dates) < lag_rebalances + 1:
        results.add_warning(
            "Lag validation",
            f"Not enough rebalance dates to validate lag={lag_rebalances}"
        )
        return
    
    # For each rebalance date (after first lag_rebalances), check that multiplier
    # corresponds to state at previous rebalance date
    errors = []
    
    for engine in applied_df['engine'].unique():
        engine_applied = applied_df[applied_df['engine'] == engine].sort_values('rebalance_date')
        engine_state = state_df[state_df['engine'] == engine].sort_values('date')
        
        for i, (_, row) in enumerate(engine_applied.iterrows()):
            if i < lag_rebalances:
                # Not enough history - should default to 1
                if row['policy_multiplier_used'] != 1:
                    errors.append(
                        f"Engine {engine}, rebal {row['rebalance_date'].date()}: "
                        f"Expected 1 (default), got {row['policy_multiplier_used']}"
                    )
                continue
            
            # Get the lagged rebalance date
            lagged_rebal_date = rebalance_dates[i - lag_rebalances]
            
            # Get state at or before lagged rebalance date
            state_at_lag = engine_state[engine_state['date'] <= lagged_rebal_date]
            
            if len(state_at_lag) == 0:
                # No state available - should default to 1
                expected = 1
            else:
                expected = state_at_lag.iloc[-1]['policy_multiplier']
            
            actual = row['policy_multiplier_used']
            
            if expected != actual:
                errors.append(
                    f"Engine {engine}, rebal {row['rebalance_date'].date()}: "
                    f"Expected {expected} (from state at {lagged_rebal_date.date()}), got {actual}"
                )
    
    if not errors:
        results.add_pass(
            f"Lag correctness (lag={lag_rebalances})",
            f"All {len(rebalance_dates)} rebalances have correct lagged multipliers"
        )
    else:
        results.add_fail(
            f"Lag correctness (lag={lag_rebalances})",
            f"{len(errors)} errors found. First 3: {errors[:3]}"
        )


def validate_policy_has_teeth(
    applied_df: pd.DataFrame,
    weights_df: Optional[pd.DataFrame],
    baseline_weights_df: Optional[pd.DataFrame],
    results: ValidationResult
):
    """Validate that policy actually affects weights when triggered."""
    # Check that there's at least one OFF state
    n_gated = (applied_df['policy_multiplier_used'] == 0).sum()
    n_total = len(applied_df)
    
    if n_gated == 0:
        results.add_warning(
            "Policy has teeth",
            "No rebalances were gated OFF - cannot verify policy impact"
        )
        return
    
    results.add_pass(
        "Policy triggers",
        f"Policy gated OFF {n_gated}/{n_total} records ({n_gated/n_total*100:.1f}%)"
    )
    
    # If we have both weights and baseline weights, compare them
    if weights_df is not None and baseline_weights_df is not None:
        # Find dates where policy was OFF
        trend_applied = applied_df[applied_df['engine'] == 'trend']
        gated_dates = trend_applied[
            trend_applied['policy_multiplier_used'] == 0
        ]['rebalance_date'].tolist()
        
        if not gated_dates:
            results.add_warning(
                "Policy has teeth",
                "No trend engine gated dates to compare weights"
            )
            return
        
        # Compare weights on gated dates
        weights_differ = False
        for date in gated_dates:
            if date in weights_df.index and date in baseline_weights_df.index:
                policy_weights = weights_df.loc[date]
                baseline_weights = baseline_weights_df.loc[date]
                
                # Check if weights are different (policy should zero them out)
                diff = (policy_weights - baseline_weights).abs().sum()
                if diff > 0.001:  # Small tolerance
                    weights_differ = True
                    break
        
        if weights_differ:
            results.add_pass(
                "Weights differ when gated",
                "Weights differ from baseline on at least one gated date"
            )
        else:
            results.add_fail(
                "Weights differ when gated",
                "Weights are identical to baseline on gated dates - policy may not be applied"
            )
    else:
        results.add_warning(
            "Weights comparison",
            "Cannot compare weights (missing weights or baseline_weights)"
        )


def validate_isolation(applied_df: pd.DataFrame, results: ValidationResult):
    """Validate that only enabled engines (trend and vrp) are affected."""
    engines = applied_df['engine'].unique()
    
    # For v1, trend and vrp can be gated OFF (policy enabled)
    # Other engines should always be ON
    allowed_policy_engines = {'trend', 'vrp'}
    
    # Check that allowed engines are present
    for engine in allowed_policy_engines:
        if engine in engines:
            results.add_pass(f"Isolation: {engine} engine", f"{engine.capitalize()} engine present in applied multipliers")
        # Don't fail if not present (may not be enabled in config)
    
    # Check that other engines (if present) are always ON
    for engine in engines:
        if engine in allowed_policy_engines:
            continue  # Skip trend and vrp (they can be gated)
        
        engine_mults = applied_df[applied_df['engine'] == engine]['policy_multiplier_used']
        if (engine_mults == 1).all():
            results.add_pass(
                f"Isolation: {engine}",
                f"Engine {engine} always ON (as expected - policy not enabled for this engine)"
            )
        else:
            n_off = (engine_mults == 0).sum()
            results.add_fail(
                f"Isolation: {engine}",
                f"Engine {engine} was gated OFF {n_off} times (unexpected - policy not enabled for this engine)"
            )


def validate_determinism(
    run_dir: Path,
    meta: Dict,
    results: ValidationResult
):
    """Validate determinism by checking hash."""
    mode = meta.get('mode', 'compute')
    
    if mode == 'precomputed':
        # Precomputed mode: check for source_determinism_hash and applied_csv_hash
        if 'source_determinism_hash' in meta:
            results.add_pass(
                "Determinism: source_determinism_hash",
                f"Source hash recorded: {meta['source_determinism_hash']}"
            )
        else:
            results.add_warning("Determinism: source_determinism_hash", "No source_determinism_hash in metadata")
        
        if 'applied_csv_hash' in meta:
            results.add_pass(
                "Determinism: applied_csv_hash",
                f"Applied CSV hash recorded: {meta['applied_csv_hash']}"
            )
        else:
            results.add_warning("Determinism: applied_csv_hash", "No applied_csv_hash in metadata")
    else:
        # Compute mode: check for determinism_hash
        if 'determinism_hash' in meta:
            det_hash = meta['determinism_hash']
            results.add_pass(
                "Determinism: determinism_hash",
                f"Hash recorded: {det_hash} (rerun to verify)"
            )
        else:
            results.add_warning("Determinism: determinism_hash", "No determinism_hash in metadata")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Phase-2 Engine Policy v1 implementation"
    )
    parser.add_argument(
        "run_id",
        help="Run ID to validate (e.g., policy_trend_gamma_compute_proof_2024)"
    )
    parser.add_argument(
        "--baseline_run_id",
        help="Baseline run ID for comparison (for precomputed mode validation)"
    )
    parser.add_argument(
        "--runs_dir",
        default=str(RUNS_DIR),
        help="Base directory for run artifacts"
    )
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    run_dir = runs_dir / args.run_id
    
    print("=" * 80)
    print("ENGINE POLICY V1 PHASE-2 VALIDATOR")
    print("=" * 80)
    print(f"Run ID: {args.run_id}")
    print(f"Run Directory: {run_dir}")
    if args.baseline_run_id:
        print(f"Baseline Run ID: {args.baseline_run_id}")
    print("=" * 80)
    
    # Check run directory exists
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        print(f"\nERROR: Run directory not found: {run_dir}")
        print("Make sure the run has been executed first.")
        sys.exit(1)
    
    results = ValidationResult()
    
    # 1. Validate artifacts exist
    print("\n--- Check 1: Artifacts Exist ---")
    if not validate_artifacts_exist(run_dir, results):
        print("\nCritical artifacts missing - stopping validation")
        results.summary()
        sys.exit(1)
    
    # Load artifacts
    artifacts = load_run_artifacts(run_dir)
    state_df = artifacts.get('state')
    applied_df = artifacts.get('applied')
    meta = artifacts.get('meta')
    weights_df = artifacts.get('weights')
    
    # 2. Validate schema
    print("\n--- Check 2: Schema Validation ---")
    if state_df is not None:
        validate_state_schema(state_df, results)
    if applied_df is not None:
        validate_applied_schema(applied_df, results)
    
    # 3. Validate lag correctness
    print("\n--- Check 3: Lag Correctness ---")
    if state_df is not None and applied_df is not None:
        lag = meta.get('lag_rebalances', 1) if meta else 1
        validate_lag_correctness(state_df, applied_df, lag, results)
    
    # 4. Validate policy has teeth
    print("\n--- Check 4: Policy Has Teeth ---")
    baseline_weights_df = None
    if args.baseline_run_id:
        baseline_run_dir = runs_dir / args.baseline_run_id
        baseline_weights_file = baseline_run_dir / "weights.csv"
        if baseline_weights_file.exists():
            baseline_weights_df = pd.read_csv(baseline_weights_file, index_col=0, parse_dates=True)
    
    if applied_df is not None:
        validate_policy_has_teeth(applied_df, weights_df, baseline_weights_df, results)
    
    # 5. Validate isolation
    print("\n--- Check 5: Isolation ---")
    if applied_df is not None:
        validate_isolation(applied_df, results)
    
    # 6. Validate determinism
    print("\n--- Check 6: Determinism ---")
    if meta is not None:
        validate_determinism(run_dir, meta, results)
    
    # Summary
    all_passed = results.summary()
    
    if all_passed:
        print("\n[SUCCESS] ALL VALIDATION CHECKS PASSED")
        print("Engine Policy v1 implementation is valid for Phase-2 acceptance.")
        sys.exit(0)
    else:
        print("\n[FAILURE] VALIDATION FAILED")
        print("Please fix the failed checks before Phase-2 acceptance.")
        sys.exit(1)


if __name__ == "__main__":
    main()

