"""
Run Phase 3B Waterfall Attribution Diagnostic

This is Step 1 (FIRST) of Phase 3B - the non-negotiable gate.

Computes stage-by-stage waterfall attribution showing where alpha disappears:
- Stages: Raw → Post-Policy → Post-RT (blue) → Post-Allocator (traded)
- Metrics: CAGR, Vol, Sharpe, MaxDD, Time-under-water
- Regime-conditioned: high-vol vs low-vol, crisis vs calm
- For aggregate portfolio AND each engine

Usage:
    python scripts/diagnostics/run_waterfall_attribution.py --run_id <run_id>
"""

import sys
import argparse
import json
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.diagnostics.waterfall_attribution import compute_waterfall_attribution, format_waterfall_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run Phase 3B Waterfall Attribution Diagnostic')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID to analyze')
    parser.add_argument('--run_dir', type=str, default=None, help='Optional path to run directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: same as run_dir)')
    
    args = parser.parse_args()
    
    run_id = args.run_id
    run_dir = Path(args.run_dir) if args.run_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if output_dir is None:
        if run_dir:
            output_dir = run_dir
        else:
            output_dir = Path(f"reports/runs/{run_id}")
    
    logger.info(f"Computing waterfall attribution for run: {run_id}")
    
    try:
        # Compute waterfall attribution
        report = compute_waterfall_attribution(run_id, run_dir)
        
        # Save JSON report
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / 'waterfall_attribution.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved JSON report: {json_path}")
        
        # Generate and save Markdown report
        md_report = format_waterfall_report(report)
        md_path = output_dir / 'waterfall_attribution.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        logger.info(f"Saved Markdown report: {md_path}")
        
        logger.info("Waterfall attribution complete")
        
    except Exception as e:
        logger.error(f"Failed to compute waterfall attribution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
