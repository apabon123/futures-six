"""
Generate Canonical Diagnostic Report

Generates comprehensive diagnostic reports in JSON and Markdown formats.

Outputs:
- canonical_diagnostics.json (machine-readable)
- canonical_diagnostics.md (human-readable)

Location: reports/runs/{run_id}/
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.diagnostics.canonical_diagnostics import (
    generate_canonical_diagnostics,
    generate_markdown_report,
)
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Canonical Diagnostic Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates comprehensive diagnostic reports including:
1. Performance Decomposition (gross/net returns, allocator drag, policy drag)
2. Engine-level Sharpe & Contribution (per meta-sleeve metrics)
3. Constraint Binding Report (allocator, policy gates, leverage caps)
4. Path Diagnostics (worst drawdowns, worst/best months)

Outputs:
- canonical_diagnostics.json (machine-readable)
- canonical_diagnostics.md (human-readable)

Examples:
  # Generate report for a run
  python scripts/diagnostics/generate_canonical_diagnostics.py --run_id canonical_frozen_stack_precomputed_20260113_100007
  
  # Generate report with custom output path
  python scripts/diagnostics/generate_canonical_diagnostics.py --run_id my_run --output_path reports/custom/
        """
    )
    
    parser.add_argument(
        '--run_id',
        type=str,
        required=True,
        help='Run ID to analyze'
    )
    
    parser.add_argument(
        '--run_dir',
        type=str,
        default=None,
        help='Optional: Custom path to run directory (default: reports/runs/{run_id})'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Optional: Custom output path for reports (default: reports/runs/{run_id}/)'
    )
    
    args = parser.parse_args()
    
    try:
        # Determine run directory
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            run_dir = Path(f"reports/runs/{args.run_id}")
        
        if not run_dir.exists():
            logger.error(f"Run directory not found: {run_dir}")
            return 1
        
        # Generate diagnostics
        logger.info(f"Generating canonical diagnostics for run: {args.run_id}")
        report = generate_canonical_diagnostics(args.run_id, run_dir=run_dir)
        
        # Determine output directory
        if args.output_path:
            output_dir = Path(args.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = run_dir
        
        # Save JSON report
        json_path = output_dir / "canonical_diagnostics.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"✓ JSON report saved to: {json_path}")
        
        # Generate and save Markdown report
        markdown_report = generate_markdown_report(report)
        md_path = output_dir / "canonical_diagnostics.md"
        with open(md_path, 'w') as f:
            f.write(markdown_report)
        logger.info(f"✓ Markdown report saved to: {md_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("CANONICAL DIAGNOSTIC REPORT SUMMARY")
        print("=" * 80)
        print(f"\nRun ID: {args.run_id}")
        print(f"Generated: {report['generated_at']}")
        
        decomp = report['performance_decomposition']
        print(f"\nPerformance Decomposition:")
        print(f"  Net Returns - CAGR: {decomp['net_returns']['cagr']:.2%}, Sharpe: {decomp['net_returns']['sharpe']:.4f}")
        if not (decomp['gross_returns']['cagr'] is None or np.isnan(decomp['gross_returns']['cagr'])):
            print(f"  Gross Returns - CAGR: {decomp['gross_returns']['cagr']:.2%}, Sharpe: {decomp['gross_returns']['sharpe']:.4f}")
        print(f"  Allocator Drag: {decomp['allocator_drag_bps']:.1f} bps/year")
        
        binding = report['constraint_binding']
        print(f"\nConstraint Binding:")
        print(f"  Allocator Active: {binding['allocator_active_pct']:.1f}%")
        print(f"  Policy Gates Trend: {binding['policy_gated_trend_pct']:.1f}%")
        print(f"  Policy Gates VRP: {binding['policy_gated_vrp_pct']:.1f}%")
        
        engine_metrics = report['engine_sharpe_contribution']
        if engine_metrics:
            print(f"\nEngine-level Metrics:")
            for sleeve, metrics in engine_metrics.items():
                print(f"  {sleeve}: Sharpe={metrics['unconditional_sharpe']:.4f}, PnL%={metrics['pct_of_total_pnl']:.1f}%")
        
        print("\n" + "=" * 80)
        print(f"Full reports:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
