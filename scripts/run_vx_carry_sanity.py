"""
CLI script for running VX Calendar Carry Sign-Only Sanity Check (Phase-0).

This script implements a deliberately simple, academic-style carry trading strategy
to verify that the VX calendar carry idea and P&L machinery are working correctly.

Strategy (Phase-0):
- Load VX curve prices (VX1, VX2, VX3)
- Compute carry signal: sign(VX_long - VX_short) or -sign(VX_long - VX_short) for carry capture
- Trade calendar spread directly: VX_Carry_t = P(VX_long) - P(VX_short)
- Daily strategy return = signal * spread_return
- Spread return = r_VX_long - r_VX_short (where r_k = pct_change(P_k))
- No vol targeting, no normalization beyond sign

Phase-0 Variants:
- Spread pairs: (VX2-VX1), (VX3-VX2)
- Sign directions: long spread, short spread (for carry capture in contango)

Pass Criteria:
- Sharpe >= 0.2 over full canonical window
- Reasonable behavior across years

Usage:
    # Single variant
    python scripts/run_vx_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --pair 2-1 --flip-sign
    
    # Sweep all variants and select canonical
    python scripts/run_vx_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --sweep
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml
import logging
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.diagnostics.vx_carry_sanity import (
    run_sign_only_vx_carry,
    compute_subperiod_stats,
    save_results,
    generate_plots,
    VX_SPREAD_PAIRS
)
from src.utils.phase_index import (
    get_sleeve_dirs,
    copy_to_latest,
    update_phase_index
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Meta-sleeve and atomic sleeve names
META_SLEEVE = "carry"
ATOMIC_SLEEVE = "vx_calendar_carry"
ATOMIC_SLEEVE_VARIANTS = "vx_calendar_carry_variants"  # For variant sweep


def parse_pair(pair_str: str) -> tuple:
    """Parse pair string like '2-1' into (2, 1) tuple."""
    try:
        parts = pair_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair_str}. Expected format: 'X-Y'")
        long_leg = int(parts[0])
        short_leg = int(parts[1])
        return (long_leg, short_leg)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid pair format: {pair_str}. Expected format: 'X-Y' (e.g., '2-1')") from e


def run_single_variant(
    db_path: str,
    spread_pair: tuple,
    flip_sign: bool,
    start_date: str,
    end_date: str,
    break_date: str,
    output_base: Path
) -> dict:
    """Run Phase-0 for a single variant."""
    long_leg, short_leg = spread_pair
    direction_label = "short" if flip_sign else "long"
    variant_label = f"VX{long_leg}-VX{short_leg}_{direction_label}"
    variant_dir_label = f"{long_leg}-{short_leg}_{direction_label}"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Running Phase-0 for variant: {variant_label}")
    logger.info(f"{'='*80}")
    
    # Run strategy
    result = run_sign_only_vx_carry(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        spread_pair=spread_pair,
        flip_sign=flip_sign
    )
    
    # Compute subperiod stats
    subperiod_stats = compute_subperiod_stats(
        portfolio_returns=result['portfolio_returns'],
        equity_curve=result['equity_curve'],
        break_date=break_date
    )
    
    # Prepare stats
    stats = {
        'portfolio': result['metrics'],
        'per_asset': result['per_asset_stats']
    }
    
    # Determine output directory
    sleeve_dirs = get_sleeve_dirs(META_SLEEVE, ATOMIC_SLEEVE_VARIANTS)
    archive_dir = sleeve_dirs["archive"] / variant_dir_label
    latest_dir = sleeve_dirs["latest"] / variant_dir_label
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp_dir = archive_dir / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_results(
        results=result,
        stats=stats,
        subperiod_stats=subperiod_stats,
        output_dir=timestamp_dir,
        start_date=start_date,
        end_date=end_date
    )
    
    # Generate plots
    generate_plots(
        results=result,
        stats=stats,
        subperiod_stats=subperiod_stats,
        output_dir=timestamp_dir
    )
    
    # Copy to latest
    copy_to_latest(timestamp_dir, latest_dir)
    
    # Extract summary metrics
    metrics = result['metrics']
    summary = {
        'variant': variant_label,
        'variant_dir': variant_dir_label,
        'spread_pair': spread_pair,
        'flip_sign': flip_sign,
        'sharpe': metrics.get('Sharpe', metrics.get('sharpe', 0)),
        'cagr': metrics.get('CAGR', metrics.get('cagr', 0)),
        'maxdd': metrics.get('MaxDD', metrics.get('maxdd', 0)),
        'hitrate': metrics.get('HitRate', metrics.get('hit_rate', 0)),
        'vol': metrics.get('Vol', metrics.get('vol', 0)),
        'n_days': metrics.get('n_days', len(result['portfolio_returns'])),
        'years': metrics.get('years', metrics.get('n_days', len(result['portfolio_returns'])) / 252),
        'start_date': result['portfolio_returns'].index[0].strftime('%Y-%m-%d'),
        'end_date': result['portfolio_returns'].index[-1].strftime('%Y-%m-%d'),
        'latest_path': str(latest_dir.relative_to(Path("reports"))),
        'collision_days_dropped': result.get('collision_days_dropped', 0)
    }
    
    logger.info(f"\n{variant_label} Results:")
    logger.info(f"  Sharpe: {summary['sharpe']:.4f}")
    logger.info(f"  CAGR: {summary['cagr']:.2%}")
    logger.info(f"  MaxDD: {summary['maxdd']:.2%}")
    logger.info(f"  HitRate: {summary['hitrate']:.2%}")
    logger.info(f"  n_days: {summary['n_days']}")
    
    return summary


def run_sweep(
    db_path: str,
    pairs: list,
    start_date: str,
    end_date: str,
    break_date: str
) -> pd.DataFrame:
    """Run Phase-0 sweep for all variants and generate summary."""
    logger.info("\n" + "="*80)
    logger.info("VX CALENDAR CARRY PHASE-0 VARIANT SWEEP")
    logger.info("="*80)
    logger.info(f"Start date: {start_date}")
    logger.info(f"End date: {end_date}")
    
    # Generate all variants: each pair × both sign directions
    variants = []
    for pair in pairs:
        variants.append((pair, False))  # Long spread
        variants.append((pair, True))   # Short spread (carry capture)
    
    variant_labels = [f'VX{lr}-VX{sr}_{"short" if flip else "long"}' for (lr, sr), flip in variants]
    logger.info(f"Variants to test: {variant_labels}")
    
    summaries = []
    
    for spread_pair, flip_sign in variants:
        try:
            summary = run_single_variant(
                db_path=db_path,
                spread_pair=spread_pair,
                flip_sign=flip_sign,
                start_date=start_date,
                end_date=end_date,
                break_date=break_date,
                output_base=None
            )
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Error running variant {spread_pair} flip={flip_sign}: {e}")
            continue
    
    if not summaries:
        raise ValueError("No variants completed successfully")
    
    # Create summary DataFrame
    df = pd.DataFrame(summaries)
    df = df.sort_values('sharpe', ascending=False)
    
    # Save summary
    sleeve_dirs = get_sleeve_dirs(META_SLEEVE, ATOMIC_SLEEVE_VARIANTS)
    summary_dir = sleeve_dirs["base"] / "summary" / "latest"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = summary_dir / "variant_summary.csv"
    df.to_csv(summary_file, index=False)
    logger.info(f"\nSaved variant summary to: {summary_file}")
    
    return df


def select_canonical(df: pd.DataFrame, min_bars: int = 1200) -> dict:
    """
    Select canonical variant based on highest Sharpe with minimum bars constraint.
    """
    # Filter by minimum bars and valid Sharpe (not NaN, not zero)
    valid = df[
        (df['n_days'] >= min_bars) & 
        (df['sharpe'].notna()) & 
        (df['sharpe'] != 0)
    ].copy()
    
    if valid.empty:
        logger.warning(f"No variants meet minimum bars requirement ({min_bars}) with valid Sharpe. Using all variants with valid Sharpe.")
        valid = df[(df['sharpe'].notna()) & (df['sharpe'] != 0)].copy()
    
    if valid.empty:
        logger.warning("No variants with valid Sharpe found. Using first variant.")
        valid = df.copy()
    
    # Select highest Sharpe
    valid_sorted = valid.sort_values('sharpe', ascending=False)
    canonical = valid_sorted.iloc[0].to_dict()
    logger.info(f"Selected canonical variant: {canonical['variant']} (Sharpe {canonical['sharpe']:.4f})")
    
    return canonical


def main():
    parser = argparse.ArgumentParser(
        description="Run VX Calendar Carry Sign-Only Sanity Check (Phase-0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single variant (VX2-VX1, short spread for carry capture)
  python scripts/run_vx_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --pair 2-1 --flip-sign
  
  # Sweep all variants and select canonical
  python scripts/run_vx_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --sweep
        """
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=CANONICAL_START,
        help=f"Start date for backtest (YYYY-MM-DD), default: {CANONICAL_START}"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date for backtest (YYYY-MM-DD), default: {CANONICAL_END}"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: reports/sanity_checks/carry/vx_calendar_carry/archive/<timestamp>)"
    )
    parser.add_argument(
        "--break_date",
        type=str,
        default="2022-01-01",
        help="Date to split subperiods for analysis (default: 2022-01-01)"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help="Path to canonical database (default: from configs/data.yaml)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data config file (default: configs/data.yaml)"
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Spread pair to test (e.g., '2-1', '3-2'). Only used if not --sweep."
    )
    parser.add_argument(
        "--flip-sign",
        action="store_true",
        help="Flip sign for carry capture (short spread in contango). Only used if not --sweep."
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run variant sweep for all spread pairs and sign directions, then select canonical"
    )
    parser.add_argument(
        "--min_bars",
        type=int,
        default=1200,
        help="Minimum number of bars required for canonical selection (default: 1200)"
    )
    
    args = parser.parse_args()
    
    try:
        # Determine DB path
        if args.db_path:
            db_path = args.db_path
        else:
            config_path = Path(args.config)
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                logger.error("Please specify --db-path or ensure configs/data.yaml exists")
                sys.exit(1)
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            db_path = config['db']['path']
        
        if args.sweep:
            # Sweep mode: run all variants
            pairs_to_test = VX_SPREAD_PAIRS.copy()
            
            # Run sweep
            df_summary = run_sweep(
                db_path=db_path,
                pairs=pairs_to_test,
                start_date=args.start,
                end_date=args.end,
                break_date=args.break_date
            )
            
            # Select canonical
            canonical = select_canonical(df_summary, min_bars=args.min_bars)
            
            # Print summary table
            print("\n" + "="*80)
            print("PHASE-0 VARIANT SWEEP SUMMARY")
            print("="*80)
            print("\nRanked by Sharpe (highest first):")
            display_cols = ['variant', 'sharpe', 'cagr', 'maxdd', 'hitrate', 'vol', 'n_days']
            print(df_summary[display_cols].to_string(index=False))
            
            print("\n" + "="*80)
            print(f"[CANONICAL] Selected variant: {canonical['variant']}")
            print("="*80)
            print(f"  Sharpe: {canonical['sharpe']:.4f}")
            print(f"  CAGR: {canonical['cagr']:.2%}")
            print(f"  MaxDD: {canonical['maxdd']:.2%}")
            print(f"  HitRate: {canonical['hitrate']:.2%}")
            print(f"  Vol: {canonical['vol']:.2%}")
            print(f"  n_days: {canonical['n_days']}")
            print(f"  Period: {canonical['start_date']} to {canonical['end_date']}")
            
            # Update phase index to canonical variant
            canonical_latest_dir = Path("reports") / canonical['latest_path']
            update_phase_index(
                META_SLEEVE,
                ATOMIC_SLEEVE_VARIANTS,
                "phase0",
                path=canonical['latest_path']
            )
            logger.info(f"\nUpdated phase_index/{META_SLEEVE}/{ATOMIC_SLEEVE_VARIANTS}/phase0.txt -> {canonical['latest_path']}")
            
            # Save canonical selection to meta.json in summary directory
            summary_dir = get_sleeve_dirs(META_SLEEVE, ATOMIC_SLEEVE_VARIANTS)["base"] / "summary" / "latest"
            canonical_meta = {
                'canonical_variant': canonical['variant'],
                'canonical_variant_dir': canonical['variant_dir'],
                'selection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'selection_criteria': f'Highest Sharpe with min_bars >= {args.min_bars}',
                'metrics': {
                    'sharpe': canonical['sharpe'],
                    'cagr': canonical['cagr'],
                    'maxdd': canonical['maxdd'],
                    'hitrate': canonical['hitrate'],
                    'vol': canonical['vol'],
                    'n_days': canonical['n_days']
                },
                'all_variants': df_summary.to_dict('records')
            }
            with open(summary_dir / "canonical_selection.json", "w") as f:
                json.dump(canonical_meta, f, indent=2)
            logger.info(f"Saved canonical selection to: {summary_dir / 'canonical_selection.json'}")
            
        elif args.pair:
            # Single variant mode
            spread_pair = parse_pair(args.pair)
            summary = run_single_variant(
                db_path=db_path,
                spread_pair=spread_pair,
                flip_sign=args.flip_sign,
                start_date=args.start,
                end_date=args.end,
                break_date=args.break_date,
                output_base=None
            )
            
            # Update phase index for this variant
            sleeve_dirs = get_sleeve_dirs(META_SLEEVE, ATOMIC_SLEEVE_VARIANTS)
            latest_dir = sleeve_dirs["latest"] / summary['variant_dir']
            update_phase_index(
                META_SLEEVE,
                ATOMIC_SLEEVE_VARIANTS,
                "phase0",
                path=f"sanity_checks/{META_SLEEVE}/{ATOMIC_SLEEVE_VARIANTS}/{summary['variant_dir']}/latest"
            )
            
        else:
            # Default: run VX2-VX1 long spread (legacy mode)
            logger.info("=" * 80)
            logger.info("VX CALENDAR CARRY SIGN-ONLY SANITY CHECK (Phase-0)")
            logger.info("=" * 80)
            logger.info(f"Start date: {args.start}")
            logger.info(f"End date: {args.end}")
            logger.info(f"Subperiod break: {args.break_date}")
            logger.info(f"Database: {db_path}")
            logger.info("Using default: VX2-VX1, long spread")
            
            # Run sign-only VX carry strategy
            logger.info("\n[2/6] Running sign-only VX calendar carry strategy...")
            result = run_sign_only_vx_carry(
                db_path=db_path,
                start_date=args.start,
                end_date=args.end,
                spread_pair=(2, 1),
                flip_sign=False
            )
            
            # Compute subperiod stats
            logger.info("\n[3/6] Computing subperiod statistics...")
            subperiod_stats = compute_subperiod_stats(
                portfolio_returns=result['portfolio_returns'],
                equity_curve=result['equity_curve'],
                break_date=args.break_date
            )
            
            # Prepare stats dict for save_results
            stats = {
                'portfolio': result['metrics'],
                'per_asset': result['per_asset_stats']
            }
            
            # Determine output directory structure
            sleeve_dirs = get_sleeve_dirs(META_SLEEVE, ATOMIC_SLEEVE)
            archive_dir = sleeve_dirs["archive"]
            latest_dir = sleeve_dirs["latest"]
            
            # Create timestamp subdirectory for this run
            if args.output_dir:
                timestamp_dir = Path(args.output_dir)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_dir.mkdir(parents=True, exist_ok=True)
                timestamp_dir = archive_dir / timestamp
            
            timestamp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            logger.info(f"\n[4/6] Saving results to {timestamp_dir}...")
            save_results(
                results=result,
                stats=stats,
                subperiod_stats=subperiod_stats,
                output_dir=timestamp_dir,
                start_date=args.start,
                end_date=args.end
            )
            
            # Generate plots
            logger.info("[5/6] Generating plots...")
            generate_plots(
                results=result,
                stats=stats,
                subperiod_stats=subperiod_stats,
                output_dir=timestamp_dir
            )
            
            # Update canonical latest/ directory
            logger.info("\n[6/6] Updating canonical latest/ directory...")
            copy_to_latest(timestamp_dir, latest_dir)
            logger.info(f"  Canonical Phase-0 results: {latest_dir}")
            
            # Update phase index
            update_phase_index(META_SLEEVE, ATOMIC_SLEEVE, "phase0")
            logger.info(f"  Updated phase_index/{META_SLEEVE}/{ATOMIC_SLEEVE}/phase0.txt")
            
            # Print results
            portfolio_sharpe = result['metrics'].get('Sharpe', result['metrics'].get('sharpe', 0))
            portfolio_cagr = result['metrics'].get('CAGR', result['metrics'].get('cagr', 0))
            portfolio_vol = result['metrics'].get('Vol', result['metrics'].get('vol', 0))
            portfolio_maxdd = result['metrics'].get('MaxDD', result['metrics'].get('maxdd', 0))
            portfolio_hitrate = result['metrics'].get('HitRate', result['metrics'].get('hit_rate', 0))
            n_days = result['metrics'].get('n_days', len(result['portfolio_returns']))
            years = result['metrics'].get('years', n_days / 252.0)
            
            print("\n" + "=" * 80)
            print("VX CALENDAR CARRY SIGN-ONLY SANITY CHECK RESULTS (Phase-0)")
            print("=" * 80)
            
            print(f"\nPortfolio:")
            print(f"  CAGR:         {portfolio_cagr:8.2%}")
            print(f"  Vol:          {portfolio_vol:8.2%}")
            print(f"  Sharpe:       {portfolio_sharpe:8.4f}")
            print(f"  MaxDD:        {portfolio_maxdd:8.2%}")
            print(f"  HitRate:      {portfolio_hitrate:8.2%}")
            print(f"  n_days:       {n_days:8d}")
            print(f"  years:        {years:8.2f}")
            
            # Print subperiod stats
            if 'pre' in subperiod_stats and 'post' in subperiod_stats:
                print(f"\nSubperiods:")
                if subperiod_stats['pre']:
                    pre_sharpe = subperiod_stats['pre'].get('Sharpe', 0)
                    pre_cagr = subperiod_stats['pre'].get('CAGR', 0)
                    print(f"  Pre-2022:     Sharpe={pre_sharpe:.4f}, CAGR={pre_cagr:.2%}")
                if subperiod_stats['post']:
                    post_sharpe = subperiod_stats['post'].get('Sharpe', 0)
                    post_cagr = subperiod_stats['post'].get('CAGR', 0)
                    print(f"  Post-2022:    Sharpe={post_sharpe:.4f}, CAGR={post_cagr:.2%}")
            
            # Phase-0 Decision Gate
            print("\n" + "=" * 80)
            if portfolio_sharpe >= 0.2:
                print("[PASS] VX calendar carry contains standalone economic edge -> proceed to Phase-1")
            else:
                print("[FAIL] VX calendar carry fails Phase-0 -> parked unless architecture changes")
            print("=" * 80)
        
        logger.info("\n" + "=" * 80)
        logger.info("Diagnostics complete!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
