#!/usr/bin/env python3
"""
Allocator State v1 Diagnostics Script

Generates allocator_state artifacts from existing backtest runs.
Computes 10 canonical state features for later regime classification and risk management.

This script:
1. Loads artifacts from an existing run_id (portfolio returns, equity curve, asset returns)
2. Optionally loads trend unit returns and sleeve returns if available
3. Computes allocator_state.csv with 10 features
4. Saves artifacts and metadata

Usage:
    python scripts/diagnostics/run_allocator_state_v1.py --run_id core_v9_baseline_phase0_20251217_193451
    python scripts/diagnostics/run_allocator_state_v1.py --run_id <run_id> --output_dir reports/runs/<run_id>
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import json

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.allocator.state_v1 import AllocatorStateV1
from src.allocator.state_validate import validate_allocator_state_v1, validate_inputs_aligned
from src.utils.canonical_window import load_canonical_window

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_run_directory(run_id: str) -> Optional[Path]:
    """
    Find the run directory for a given run_id.
    
    Args:
        run_id: Run identifier (e.g., 'core_v9_baseline_phase0_20251217_193451')
    
    Returns:
        Path to run directory, or None if not found
    """
    runs_dir = Path("reports/runs")
    
    if not runs_dir.exists():
        logger.error(f"Runs directory not found: {runs_dir}")
        return None
    
    # Try exact match first
    exact_match = runs_dir / run_id
    if exact_match.exists() and exact_match.is_dir():
        return exact_match
    
    # Try partial match
    for d in runs_dir.iterdir():
        if d.is_dir() and run_id in d.name:
            logger.info(f"Found run directory: {d.name}")
            return d
    
    logger.error(f"Run directory not found for: {run_id}")
    return None


def load_run_artifacts(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load artifacts from a run directory.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        Dict with keys: 'portfolio_returns', 'equity_curve', 'asset_returns',
        'trend_unit_returns' (optional), 'sleeve_returns' (optional)
    """
    artifacts = {}
    
    # Load portfolio returns (required)
    portfolio_returns_file = run_dir / "portfolio_returns.csv"
    if not portfolio_returns_file.exists():
        raise FileNotFoundError(f"Portfolio returns not found: {portfolio_returns_file}")
    
    df = pd.read_csv(portfolio_returns_file, index_col=0, parse_dates=True)
    if 'ret' in df.columns:
        artifacts['portfolio_returns'] = df['ret']
    else:
        artifacts['portfolio_returns'] = df.iloc[:, 0]
    
    logger.info(f"Loaded portfolio_returns: {len(artifacts['portfolio_returns'])} rows")
    
    # Load equity curve (required)
    equity_curve_file = run_dir / "equity_curve.csv"
    if not equity_curve_file.exists():
        raise FileNotFoundError(f"Equity curve not found: {equity_curve_file}")
    
    df = pd.read_csv(equity_curve_file, index_col=0, parse_dates=True)
    if 'equity' in df.columns:
        artifacts['equity_curve'] = df['equity']
    else:
        artifacts['equity_curve'] = df.iloc[:, 0]
    
    logger.info(f"Loaded equity_curve: {len(artifacts['equity_curve'])} rows")
    
    # Load asset returns (required)
    asset_returns_file = run_dir / "asset_returns.csv"
    if not asset_returns_file.exists():
        raise FileNotFoundError(f"Asset returns not found: {asset_returns_file}")
    
    artifacts['asset_returns'] = pd.read_csv(asset_returns_file, index_col=0, parse_dates=True)
    logger.info(
        f"Loaded asset_returns: {len(artifacts['asset_returns'])} rows, "
        f"{len(artifacts['asset_returns'].columns)} assets"
    )
    
    # Load trend unit returns (optional - Stage 4A)
    trend_unit_returns_file = run_dir / "trend_unit_returns.csv"
    if trend_unit_returns_file.exists():
        try:
            artifacts['trend_unit_returns'] = pd.read_csv(trend_unit_returns_file, index_col=0, parse_dates=True)
            logger.info(
                f"Loaded trend_unit_returns: {len(artifacts['trend_unit_returns'])} rows, "
                f"{len(artifacts['trend_unit_returns'].columns)} assets"
            )
        except Exception as e:
            logger.warning(f"Failed to load trend_unit_returns: {e}")
            artifacts['trend_unit_returns'] = None
    else:
        logger.info("trend_unit_returns.csv not found (optional - will be available after Stage 4A)")
        artifacts['trend_unit_returns'] = None
    
    # Load sleeve returns (optional - Stage 4A)
    sleeve_returns_file = run_dir / "sleeve_returns.csv"
    if sleeve_returns_file.exists():
        try:
            artifacts['sleeve_returns'] = pd.read_csv(sleeve_returns_file, index_col=0, parse_dates=True)
            logger.info(
                f"Loaded sleeve_returns: {len(artifacts['sleeve_returns'])} rows, "
                f"{len(artifacts['sleeve_returns'].columns)} sleeves"
            )
        except Exception as e:
            logger.warning(f"Failed to load sleeve_returns: {e}")
            artifacts['sleeve_returns'] = None
    else:
        logger.info("sleeve_returns.csv not found (optional - will be available after Stage 4A)")
        artifacts['sleeve_returns'] = None
    
    return artifacts


def save_allocator_state(
    state: pd.DataFrame,
    run_dir: Path,
    requested_start: Optional[str] = None,
    requested_end: Optional[str] = None
) -> None:
    """
    Save allocator state artifacts and metadata.
    
    Args:
        state: Allocator state DataFrame
        run_dir: Path to run directory
        requested_start: Requested start date (for logging)
        requested_end: Requested end date (for logging)
    """
    from src.allocator.state_v1 import LOOKBACKS, REQUIRED_FEATURES, OPTIONAL_FEATURES
    
    # Save allocator_state.csv
    output_file = run_dir / "allocator_state_v1.csv"
    state.to_csv(output_file)
    logger.info(f"Saved allocator_state_v1.csv: {output_file}")
    
    # Extract metadata from state.attrs (set by AllocatorStateV1.compute)
    features_present = state.attrs.get('features_present', list(state.columns))
    features_missing = state.attrs.get('features_missing', [])
    rows_before = state.attrs.get('rows_before_dropna', len(state))
    rows_after = state.attrs.get('rows_after_dropna', len(state))
    rows_dropped = state.attrs.get('rows_dropped', 0)
    
    effective_start = state.index[0].strftime('%Y-%m-%d') if len(state) > 0 else None
    effective_end = state.index[-1].strftime('%Y-%m-%d') if len(state) > 0 else None
    
    # Compute effective start shift
    effective_start_shift_days = 0
    if requested_start and effective_start:
        requested_start_dt = pd.Timestamp(requested_start)
        effective_start_dt = pd.Timestamp(effective_start)
        effective_start_shift_days = (effective_start_dt - requested_start_dt).days
        
        if effective_start_shift_days > 0:
            logger.info(
                f"Requested start: {requested_start}, Effective start: {effective_start} "
                f"(+{effective_start_shift_days} days due to rolling windows)"
            )
        else:
            logger.info(f"Effective start: {effective_start}")
    
    # Compute requested rows (if we had data from requested_start)
    rows_requested = len(state) + rows_dropped if rows_dropped > 0 else len(state)
    
    # Create metadata with canonical structure
    meta = {
        'allocator_state_version': AllocatorStateV1.VERSION,
        'lookbacks': LOOKBACKS,
        'required_features': REQUIRED_FEATURES,
        'optional_features': OPTIONAL_FEATURES,
        'features_present': features_present,
        'features_missing': features_missing,
        'rows_requested': rows_requested,
        'rows_valid': rows_after,
        'rows_dropped': rows_dropped,
        'effective_start_date': effective_start,
        'effective_end_date': effective_end,
        'requested_start_date': requested_start,
        'requested_end_date': requested_end,
        'effective_start_shift_days': effective_start_shift_days,
        'generated_at': datetime.now().isoformat()
    }
    
    # Save metadata
    meta_file = run_dir / "allocator_state_v1_meta.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Saved allocator_state_v1_meta.json: {meta_file}")
    
    # Log summary
    logger.info("=" * 80)
    logger.info("Allocator State v1 Summary")
    logger.info("=" * 80)
    logger.info(f"Version: {meta['allocator_state_version']}")
    logger.info(f"Rows requested: {meta['rows_requested']}")
    logger.info(f"Rows valid: {meta['rows_valid']}")
    logger.info(f"Rows dropped: {meta['rows_dropped']}")
    logger.info(f"Effective date range: {effective_start} to {effective_end}")
    logger.info(f"Effective start shift: +{meta['effective_start_shift_days']} days")
    logger.info(f"Features present ({len(features_present)}): {', '.join(features_present)}")
    if features_missing:
        logger.info(f"Features missing ({len(features_missing)}): {', '.join(features_missing)}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Allocator State v1 artifacts from existing backtest run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with run_id
  python scripts/diagnostics/run_allocator_state_v1.py --run_id core_v9_baseline_phase0_20251217_193451
  
  # Specify custom output directory
  python scripts/diagnostics/run_allocator_state_v1.py --run_id <run_id> --output_dir reports/runs/<run_id>
        """
    )
    
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run identifier (e.g., 'core_v9_baseline_phase0_20251217_193451')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (default: same as run directory)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Allocator State v1 Generator")
    logger.info("=" * 80)
    logger.info(f"Run ID: {args.run_id}")
    
    # Find run directory
    run_dir = find_run_directory(args.run_id)
    if run_dir is None:
        logger.error(f"Run directory not found for: {args.run_id}")
        sys.exit(1)
    
    # Override output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = run_dir
    
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load artifacts
    logger.info("\nLoading artifacts...")
    try:
        artifacts = load_run_artifacts(run_dir)
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load canonical window for logging
    try:
        canonical_start, canonical_end = load_canonical_window()
    except Exception as e:
        logger.warning(f"Could not load canonical window: {e}")
        canonical_start, canonical_end = None, None
    
    # Validate input alignment
    logger.info("\nValidating input alignment...")
    try:
        validate_inputs_aligned(
            portfolio_returns=artifacts['portfolio_returns'],
            equity_curve=artifacts['equity_curve'],
            asset_returns=artifacts['asset_returns']
        )
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Compute allocator state
    logger.info("\nComputing allocator state features...")
    try:
        allocator = AllocatorStateV1()
        state = allocator.compute(
            portfolio_returns=artifacts['portfolio_returns'],
            equity_curve=artifacts['equity_curve'],
            asset_returns=artifacts['asset_returns'],
            trend_unit_returns=artifacts.get('trend_unit_returns'),
            sleeve_returns=artifacts.get('sleeve_returns')
        )
    except Exception as e:
        logger.error(f"Failed to compute allocator state: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check for empty state
    if len(state) == 0:
        logger.error("Allocator state is empty after computation")
        sys.exit(1)
    
    # Validate allocator state
    logger.info("\nValidating allocator state...")
    try:
        # Create preliminary metadata for validation
        prelim_meta = {
            'rows_dropped': state.attrs.get('rows_dropped', 0),
            'rows_requested': state.attrs.get('rows_before_dropna', len(state))
        }
        validate_allocator_state_v1(state, prelim_meta)
    except Exception as e:
        logger.error(f"Allocator state validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save artifacts
    logger.info("\nSaving allocator state artifacts...")
    try:
        save_allocator_state(
            state=state,
            run_dir=output_dir,
            requested_start=canonical_start,
            requested_end=canonical_end
        )
    except Exception as e:
        logger.error(f"Failed to save allocator state: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Quick inspection
    logger.info("\nQuick inspection:")
    logger.info(f"Head (first 5 rows):")
    print(state.head())
    logger.info(f"\nTail (last 5 rows):")
    print(state.tail())
    logger.info(f"\nSummary statistics:")
    print(state.describe())
    
    logger.info("\n✓ Allocator state v1 generation complete")


if __name__ == "__main__":
    main()

