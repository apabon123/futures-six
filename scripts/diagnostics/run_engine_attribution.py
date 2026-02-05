"""
Run Engine Attribution at Post-Construction

This script computes engine-level contribution analysis at the Post-Construction stage,
which is the canonical system belief evaluation layer.

This is part of the Phase 3B diagnostics framework.

Usage:
    # Single run
    python scripts/diagnostics/run_engine_attribution.py --run_id <run_id>
    
    # A/B test (compare two runs)
    python scripts/diagnostics/run_engine_attribution.py --base_run_id <base> --candidate_run_id <candidate>

Outputs:
    For single run:
    - engine_attribution_post_construction.json
    - engine_attribution_post_construction.md
    
    For A/B test:
    - construction_ab_test.json
    - construction_ab_test.md

Reference:
- SYSTEM_CONSTRUCTION.md § "Portfolio Construction v1 (Canonical)"
- PROCEDURES.md § "Construction Harness Contract"
"""

import sys
import argparse
import json
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.diagnostics.engine_attribution import (
    compute_engine_attribution_post_construction,
    format_engine_attribution_report
)
from src.diagnostics.construction_ab_test import (
    portfolio_construction_ab_test,
    format_ab_test_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_attribution(run_id: str, run_dir: Path = None, output_dir: Path = None):
    """Run engine attribution for a single run."""
    if run_dir is None:
        run_dir = Path(f"reports/runs/{run_id}")
    
    if output_dir is None:
        output_dir = run_dir
    
    logger.info(f"Computing engine attribution at Post-Construction for run: {run_id}")
    
    try:
        # Compute engine attribution
        report = compute_engine_attribution_post_construction(run_id, run_dir)
        
        # Save JSON report
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / 'engine_attribution_post_construction.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved JSON report: {json_path}")
        
        # Generate and save Markdown report
        md_report = format_engine_attribution_report(report)
        md_path = output_dir / 'engine_attribution_post_construction.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        logger.info(f"Saved Markdown report: {md_path}")
        
        # Print summary
        sleeve_metrics = report.get('sleeve_metrics', {})
        sleeve_roles = report.get('sleeve_roles', {})
        
        print("\n" + "=" * 60)
        print("ENGINE ATTRIBUTION SUMMARY (Post-Construction)")
        print("=" * 60)
        print(f"Run ID: {run_id}")
        print(f"Evaluation Start: {report.get('evaluation_start', 'N/A')}")
        print(f"Sleeves Analyzed: {len(sleeve_metrics)}")
        print()
        
        # Top contributors
        top_positive = sleeve_roles.get('top_positive_contributors', [])[:3]
        top_negative = sleeve_roles.get('top_negative_contributors', [])[:3]
        diversifiers = sleeve_roles.get('diversifiers', [])
        red_flags = sleeve_roles.get('red_flags', [])
        
        print("TOP POSITIVE CONTRIBUTORS:")
        for sleeve in top_positive:
            metrics = sleeve_metrics.get(sleeve, {})
            contrib = metrics.get('total_contribution_pct')
            contrib_str = f"{contrib:.1%}" if contrib is not None else "N/A"
            print(f"  - {sleeve}: {contrib_str}")
        
        print("\nTOP NEGATIVE CONTRIBUTORS:")
        for sleeve in top_negative:
            metrics = sleeve_metrics.get(sleeve, {})
            contrib = metrics.get('total_contribution_pct')
            contrib_str = f"{contrib:.1%}" if contrib is not None else "N/A"
            print(f"  - {sleeve}: {contrib_str}")
        
        if diversifiers:
            print(f"\nDIVERSIFIERS: {', '.join(diversifiers)}")
        
        if red_flags:
            print(f"\nRED FLAGS: {', '.join(red_flags)} [INVESTIGATE]")
        
        print()
        print(f"Full report: {md_path}")
        print("=" * 60)
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to compute engine attribution: {e}", exc_info=True)
        sys.exit(1)


def run_ab_test(base_run_id: str, candidate_run_id: str, output_dir: Path = None):
    """Run A/B test comparing two runs at Post-Construction."""
    if output_dir is None:
        output_dir = Path(f"reports/runs/{candidate_run_id}")
    
    logger.info(f"Running A/B test: {base_run_id} vs {candidate_run_id}")
    
    try:
        # Run A/B test
        report = portfolio_construction_ab_test(base_run_id, candidate_run_id)
        
        # Save JSON report
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / 'construction_ab_test.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved JSON report: {json_path}")
        
        # Generate and save Markdown report
        md_report = format_ab_test_report(report)
        md_path = output_dir / 'construction_ab_test.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        logger.info(f"Saved Markdown report: {md_path}")
        
        # Print summary
        recommendation = report.get('recommendation', {})
        deltas = report.get('deltas', {})
        
        print("\n" + "=" * 60)
        print("A/B TEST SUMMARY (Post-Construction)")
        print("=" * 60)
        print(f"Base Run: {base_run_id}")
        print(f"Candidate Run: {candidate_run_id}")
        print()
        
        rec_text = recommendation.get('recommendation', 'UNKNOWN')
        rec_reason = recommendation.get('reason', '')
        
        print(f"RECOMMENDATION: {rec_text}")
        print(f"Reason: {rec_reason}")
        print()
        
        print("METRIC DELTAS:")
        for key, value in deltas.items():
            if value is not None:
                if 'sharpe' in key.lower():
                    print(f"  {key}: {value:+.3f}")
                else:
                    print(f"  {key}: {value:+.2%}")
        
        print()
        
        positives = recommendation.get('positives', [])
        issues = recommendation.get('issues', [])
        
        if positives:
            print("POSITIVES:")
            for p in positives:
                print(f"  + {p}")
        
        if issues:
            print("ISSUES:")
            for i in issues:
                print(f"  - {i}")
        
        print()
        print(f"Full report: {md_path}")
        print("=" * 60)
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to run A/B test: {e}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run Engine Attribution at Post-Construction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single run attribution
    python scripts/diagnostics/run_engine_attribution.py --run_id phase3b_baseline_traded_20260120_093953
    
    # A/B test (compare base vs candidate)
    python scripts/diagnostics/run_engine_attribution.py \\
        --base_run_id phase3b_baseline_traded_20260120_093953 \\
        --candidate_run_id my_candidate_run
"""
    )
    
    # Single run arguments
    parser.add_argument('--run_id', type=str, help='Run ID to analyze (for single run attribution)')
    parser.add_argument('--run_dir', type=str, default=None, help='Optional path to run directory')
    
    # A/B test arguments
    parser.add_argument('--base_run_id', type=str, help='Base run ID (for A/B test)')
    parser.add_argument('--candidate_run_id', type=str, help='Candidate run ID (for A/B test)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: same as run_dir)')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.base_run_id and args.candidate_run_id:
        # A/B test mode
        output_dir = Path(args.output_dir) if args.output_dir else None
        run_ab_test(args.base_run_id, args.candidate_run_id, output_dir)
    elif args.run_id:
        # Single run mode
        run_dir = Path(args.run_dir) if args.run_dir else None
        output_dir = Path(args.output_dir) if args.output_dir else None
        run_single_attribution(args.run_id, run_dir, output_dir)
    else:
        parser.error("Must provide either --run_id (for single run) or both --base_run_id and --candidate_run_id (for A/B test)")


if __name__ == '__main__':
    main()
