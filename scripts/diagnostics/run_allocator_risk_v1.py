#!/usr/bin/env python3
"""
Allocator Risk v1 Diagnostics Script

Generates risk scalars from regime classifications.

Usage:
    python scripts/diagnostics/run_allocator_risk_v1.py --run_id <run_id>
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.allocator.risk_v1 import RiskTransformerV1, DEFAULT_REGIME_SCALARS, RISK_MIN, RISK_MAX

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_run_directory(run_id: str):
    """Find the run directory for a given run_id."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate Allocator Risk v1 scalars from regime classifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with run_id
  python scripts/diagnostics/run_allocator_risk_v1.py --run_id test_stage4a_all10
        """
    )
    
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run identifier"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Allocator Risk v1 Generator")
    logger.info("=" * 80)
    logger.info(f"Run ID: {args.run_id}")
    
    # Find run directory
    run_dir = find_run_directory(args.run_id)
    if run_dir is None:
        logger.error(f"Run directory not found for: {args.run_id}")
        sys.exit(1)
    
    logger.info(f"Run directory: {run_dir}")
    
    # Load allocator state
    logger.info("\nLoading allocator state...")
    state_file = run_dir / "allocator_state_v1.csv"
    if not state_file.exists():
        logger.error(f"Allocator state not found: {state_file}")
        logger.info("Please run run_allocator_state_v1.py first")
        sys.exit(1)
    
    state_df = pd.read_csv(state_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded allocator state: {len(state_df)} rows, {len(state_df.columns)} features")
    
    # Load regime
    logger.info("\nLoading regime...")
    regime_file = run_dir / "allocator_regime_v1.csv"
    if not regime_file.exists():
        logger.error(f"Regime not found: {regime_file}")
        logger.info("Please run run_allocator_regime_v1.py first")
        sys.exit(1)
    
    regime_df = pd.read_csv(regime_file, index_col=0, parse_dates=True)
    regime = regime_df['regime']
    logger.info(f"Loaded regime: {len(regime)} rows")
    
    # Transform to risk scalars
    logger.info("\nTransforming regime to risk scalars...")
    transformer = RiskTransformerV1()
    risk_scalars = transformer.transform(state_df, regime)
    
    if risk_scalars.empty:
        logger.error("Risk transformation returned empty DataFrame")
        sys.exit(1)
    
    # Compute statistics
    risk_scalar = risk_scalars['risk_scalar']
    risk_stats = {
        'mean': float(risk_scalar.mean()),
        'std': float(risk_scalar.std()),
        'min': float(risk_scalar.min()),
        'max': float(risk_scalar.max()),
        'median': float(risk_scalar.median()),
        'q25': float(risk_scalar.quantile(0.25)),
        'q75': float(risk_scalar.quantile(0.75))
    }
    
    # Compute risk scalar by regime
    aligned_regime = regime.reindex(risk_scalar.index)
    risk_by_regime = {}
    for regime_name in DEFAULT_REGIME_SCALARS.keys():
        regime_mask = aligned_regime == regime_name
        if regime_mask.any():
            risk_by_regime[regime_name] = {
                'mean': float(risk_scalar[regime_mask].mean()),
                'min': float(risk_scalar[regime_mask].min()),
                'max': float(risk_scalar[regime_mask].max()),
                'count': int(regime_mask.sum())
            }
    
    # Save risk scalars
    risk_scalars.to_csv(run_dir / 'allocator_risk_v1.csv')
    logger.info(f"Saved allocator_risk_v1.csv: {run_dir / 'allocator_risk_v1.csv'}")
    
    # Save metadata
    meta = {
        'version': RiskTransformerV1.VERSION,
        'regime_scalar_mapping': DEFAULT_REGIME_SCALARS,
        'smoothing_alpha': transformer.smoothing_alpha,
        'smoothing_half_life': transformer.smoothing_half_life,
        'risk_bounds': [RISK_MIN, RISK_MAX],
        'risk_scalar_stats': risk_stats,
        'risk_scalar_by_regime': risk_by_regime,
        'effective_start_date': risk_scalar.index[0].strftime('%Y-%m-%d'),
        'effective_end_date': risk_scalar.index[-1].strftime('%Y-%m-%d'),
        'generated_at': datetime.now().isoformat()
    }
    
    with open(run_dir / 'allocator_risk_v1_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Saved allocator_risk_v1_meta.json: {run_dir / 'allocator_risk_v1_meta.json'}")
    
    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("Allocator Risk v1 Summary")
    logger.info("=" * 80)
    logger.info(f"Version: {meta['version']}")
    logger.info(f"Date range: {meta['effective_start_date']} to {meta['effective_end_date']}")
    logger.info(f"\nRisk Scalar Statistics:")
    logger.info(f"  Mean:   {risk_stats['mean']:.3f}")
    logger.info(f"  Median: {risk_stats['median']:.3f}")
    logger.info(f"  Std:    {risk_stats['std']:.3f}")
    logger.info(f"  Min:    {risk_stats['min']:.3f}")
    logger.info(f"  Max:    {risk_stats['max']:.3f}")
    
    logger.info(f"\nRisk Scalar by Regime:")
    for regime_name, stats in risk_by_regime.items():
        logger.info(
            f"  {regime_name:10s}: mean={stats['mean']:.3f}, "
            f"min={stats['min']:.3f}, max={stats['max']:.3f}, "
            f"n={stats['count']}"
        )
    
    logger.info("=" * 80)
    logger.info("\n✓ Allocator risk v1 generation complete")


if __name__ == "__main__":
    main()

