#!/usr/bin/env python3
"""
Allocator Regime v1 Diagnostics Script

Generates regime classifications from existing allocator state artifacts.

Usage:
    python scripts/diagnostics/run_allocator_regime_v1.py --run_id <run_id>
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.allocator.regime_v1 import RegimeClassifierV1
from src.allocator.regime_rules_v1 import get_default_thresholds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_run_directory(run_id: str) -> Optional[Path]:
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
        description="Generate Allocator Regime v1 classifications from allocator state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with run_id
  python scripts/diagnostics/run_allocator_regime_v1.py --run_id test_stage4a_all10
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
    logger.info("Allocator Regime v1 Generator")
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
    
    # Classify regime
    logger.info("\nClassifying regime...")
    classifier = RegimeClassifierV1()
    regime = classifier.classify(state_df)
    
    if regime.empty:
        logger.error("Regime classification returned empty series")
        sys.exit(1)
    
    # Compute regime statistics
    regime_counts = regime.value_counts().sort_index()
    regime_pct = (regime_counts / len(regime) * 100).round(1)
    
    # Compute transition matrix
    transitions = {}
    for i in range(len(regime) - 1):
        from_regime = regime.iloc[i]
        to_regime = regime.iloc[i + 1]
        key = f"{from_regime}->{to_regime}"
        transitions[key] = transitions.get(key, 0) + 1
    
    # Compute max consecutive days per regime
    max_consecutive = {}
    current_regime = None
    current_count = 0
    
    for r in regime:
        if r == current_regime:
            current_count += 1
        else:
            if current_regime is not None:
                max_consecutive[current_regime] = max(
                    max_consecutive.get(current_regime, 0),
                    current_count
                )
            current_regime = r
            current_count = 1
    
    # Final regime
    if current_regime is not None:
        max_consecutive[current_regime] = max(
            max_consecutive.get(current_regime, 0),
            current_count
        )
    
    # Save regime series
    regime_df = regime.to_frame('regime')
    regime_df.to_csv(run_dir / 'allocator_regime_v1.csv')
    logger.info(f"Saved allocator_regime_v1.csv: {run_dir / 'allocator_regime_v1.csv'}")
    
    # Save metadata
    meta = {
        'version': RegimeClassifierV1.VERSION,
        'thresholds': get_default_thresholds(),
        'regime_day_counts': regime_counts.to_dict(),
        'regime_percentages': regime_pct.to_dict(),
        'transition_counts': transitions,
        'max_consecutive_days': max_consecutive,
        'effective_start_date': regime.index[0].strftime('%Y-%m-%d'),
        'effective_end_date': regime.index[-1].strftime('%Y-%m-%d'),
        'generated_at': datetime.now().isoformat()
    }
    
    with open(run_dir / 'allocator_regime_v1_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Saved allocator_regime_v1_meta.json: {run_dir / 'allocator_regime_v1_meta.json'}")
    
    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("Allocator Regime v1 Summary")
    logger.info("=" * 80)
    logger.info(f"Version: {meta['version']}")
    logger.info(f"Date range: {meta['effective_start_date']} to {meta['effective_end_date']}")
    logger.info(f"\nRegime Distribution:")
    for regime_name, count in regime_counts.items():
        pct = regime_pct[regime_name]
        max_consec = max_consecutive.get(regime_name, 0)
        logger.info(f"  {regime_name:10s}: {count:4d} days ({pct:5.1f}%), max consecutive: {max_consec} days")
    
    logger.info(f"\nTop Transitions:")
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    for trans, count in sorted_transitions[:10]:
        logger.info(f"  {trans:25s}: {count:3d}")
    
    logger.info("=" * 80)
    logger.info("\n✓ Allocator regime v1 generation complete")


if __name__ == "__main__":
    main()

