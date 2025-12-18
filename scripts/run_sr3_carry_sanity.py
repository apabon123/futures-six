"""
CLI script for running SR3 Calendar Carry Sign-Only Sanity Check (Phase-0).

This script implements a deliberately simple, academic-style carry trading strategy
to verify that the SR3 calendar carry idea and P&L machinery are working correctly.

Strategy (Phase-0):
- Use ranks 1-4 only (canonical Phase-0 ranks: all start at 2020-01-02, full coverage)
- Compute carry signal: sign(RANK_2 - RANK_1) in rate space
  - r_k = 100 - P_k (convert prices to rates)
  - carry_raw = r2 - r1
  - signal = sign(carry_raw) → +1 (positive carry, long), -1 (negative carry, short), 0 (flat)
- Trade calendar spread directly (RANK_2 - RANK_1) based on this signal
- Daily strategy return = signal * spread_return
- No vol targeting, no normalization beyond sign

Pass Criteria:
- Sharpe >= 0.2 over full window
- Reasonable behavior across years

Usage:
    # Single pair
    python scripts/run_sr3_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --pair 2-1
    
    # Sweep all pairs and select canonical
    python scripts/run_sr3_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --sweep
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.agents import MarketData
from src.diagnostics.sr3_carry_sanity import (
    run_sign_only_sr3_carry,
    compute_subperiod_stats,
    save_results,
    generate_plots,
    ADJACENT_PAIRS
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
ATOMIC_SLEEVE = "sr3_calendar_carry"
ATOMIC_SLEEVE_ADJACENT = "sr3_calendar_carry_adjacent"  # For adjacent pair variants


def parse_pair(pair_str: str) -> tuple:
    """Parse pair string like '2-1' into (2, 1) tuple."""
    try:
        parts = pair_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair_str}. Expected format: 'X-Y'")
        long_rank = int(parts[0])
        short_rank = int(parts[1])
        return (long_rank, short_rank)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid pair format: {pair_str}. Expected format: 'X-Y' (e.g., '2-1')") from e


def compute_correlations(
    market,
    pair: tuple,
    spread_returns: pd.Series,
    common_dates: pd.DatetimeIndex
) -> dict:
    """Compute correlations between spread returns and rank returns."""
    long_rank, short_rank = pair
    correlations = {}
    
    try:
        # Get rank 0 returns if available
        prices_cont = market.prices_cont
        if "SR3_FRONT_CALENDAR" in prices_cont.columns:
            rank0_prices = prices_cont["SR3_FRONT_CALENDAR"]
            rank0_returns = rank0_prices.pct_change(fill_method=None).dropna()
            
            # Align dates - use intersection of both indices
            common_rank0 = spread_returns.index.intersection(rank0_returns.index)
            if len(common_rank0) > 10:  # Need sufficient overlap
                spread_aligned = spread_returns.loc[common_rank0].dropna()
                rank0_aligned = rank0_returns.loc[common_rank0].dropna()
                # Final alignment on dates present in both
                final_dates = spread_aligned.index.intersection(rank0_aligned.index)
                if len(final_dates) > 10:
                    spread_final = spread_aligned.loc[final_dates]
                    rank0_final = rank0_aligned.loc[final_dates]
                    # Check for non-zero variance
                    if spread_final.std() > 1e-10 and rank0_final.std() > 1e-10:
                        corr_spread_rank0 = spread_final.corr(rank0_final)
                        correlations['corr_spread_rank0'] = float(corr_spread_rank0) if not pd.isna(corr_spread_rank0) else None
                    else:
                        logger.warning(f"Zero variance in spread or rank0 returns for correlation")
                        correlations['corr_spread_rank0'] = None
                else:
                    correlations['corr_spread_rank0'] = None
            else:
                correlations['corr_spread_rank0'] = None
        else:
            correlations['corr_spread_rank0'] = None
    except Exception as e:
        logger.warning(f"Could not compute corr(spread, rank0): {e}")
        correlations['corr_spread_rank0'] = None
    
    try:
        # Get rank 1 returns from contract data
        # For R1-R0, rank1 is the long rank, not short_rank
        rank_to_get = long_rank if short_rank == 0 else short_rank
        close = market.get_contracts_by_root(
            root="SR3",
            ranks=[rank_to_get],
            fields=("close",),
            start=None,
            end=None
        )
        if rank_to_get in close.columns:
            rank1_prices = close[rank_to_get]
            rank1_returns = rank1_prices.pct_change(fill_method=None).dropna()
            
            # Align dates - use intersection of both indices
            common_rank1 = spread_returns.index.intersection(rank1_returns.index)
            if len(common_rank1) > 10:  # Need sufficient overlap
                spread_aligned = spread_returns.loc[common_rank1].dropna()
                rank1_aligned = rank1_returns.loc[common_rank1].dropna()
                # Final alignment on dates present in both
                final_dates = spread_aligned.index.intersection(rank1_aligned.index)
                if len(final_dates) > 10:
                    spread_final = spread_aligned.loc[final_dates]
                    rank1_final = rank1_aligned.loc[final_dates]
                    # Check for non-zero variance
                    if spread_final.std() > 1e-10 and rank1_final.std() > 1e-10:
                        corr_spread_rank1 = spread_final.corr(rank1_final)
                        correlations['corr_spread_rank1'] = float(corr_spread_rank1) if not pd.isna(corr_spread_rank1) else None
                    else:
                        logger.warning(f"Zero variance in spread or rank1 returns for correlation")
                        correlations['corr_spread_rank1'] = None
                else:
                    correlations['corr_spread_rank1'] = None
            else:
                correlations['corr_spread_rank1'] = None
        else:
            correlations['corr_spread_rank1'] = None
    except Exception as e:
        logger.warning(f"Could not compute corr(spread, rank1): {e}")
        correlations['corr_spread_rank1'] = None
    
    return correlations


def run_single_pair(
    market,
    pair: tuple,
    start_date: str,
    end_date: str,
    break_date: str,
    output_base: Path
) -> dict:
    """Run Phase-0 for a single adjacent pair."""
    long_rank, short_rank = pair
    pair_label = f"R{long_rank}-R{short_rank}"
    pair_dir_label = f"{long_rank}-{short_rank}"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Running Phase-0 for pair: {pair_label}")
    logger.info(f"{'='*80}")
    
    # Run strategy
    result = run_sign_only_sr3_carry(
        market=market,
        start_date=start_date,
        end_date=end_date,
        variant="spread",
        rank_pair=pair
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
    sleeve_dirs = get_sleeve_dirs(META_SLEEVE, ATOMIC_SLEEVE_ADJACENT)
    archive_dir = sleeve_dirs["archive"] / pair_dir_label
    latest_dir = sleeve_dirs["latest"] / pair_dir_label
    
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
        end_date=end_date,
        variant="spread"
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
    
    # Compute correlations (only for R1-R0 pair)
    correlations = {}
    if pair == (1, 0):
        # Get raw spread returns from result (stored during computation)
        spread_returns = result.get('raw_spread_returns')
        if spread_returns is not None and len(spread_returns.dropna()) > 10:
            spread_returns_clean = spread_returns.dropna()
            common_dates = spread_returns_clean.index
            correlations = compute_correlations(market, pair, spread_returns_clean, common_dates)
            logger.info(f"  Correlations computed: corr(spread, rank0)={correlations.get('corr_spread_rank0')}, corr(spread, rank1)={correlations.get('corr_spread_rank1')}")
        else:
            logger.warning(f"Insufficient spread returns data for correlation calculation")
            correlations = {'corr_spread_rank0': None, 'corr_spread_rank1': None}
    else:
        correlations = {'corr_spread_rank0': None, 'corr_spread_rank1': None}
    
    # Extract summary metrics
    metrics = result['metrics']
    summary = {
        'pair': pair_label,
        'pair_dir': pair_dir_label,
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
        'corr_spread_rank0': correlations.get('corr_spread_rank0'),
        'corr_spread_rank1': correlations.get('corr_spread_rank1')
    }
    
    logger.info(f"\n{pair_label} Results:")
    logger.info(f"  Sharpe: {summary['sharpe']:.4f}")
    logger.info(f"  CAGR: {summary['cagr']:.2%}")
    logger.info(f"  MaxDD: {summary['maxdd']:.2%}")
    logger.info(f"  HitRate: {summary['hitrate']:.2%}")
    logger.info(f"  n_days: {summary['n_days']}")
    
    return summary


def run_sweep(
    market,
    pairs: list,
    start_date: str,
    end_date: str,
    break_date: str
) -> pd.DataFrame:
    """Run Phase-0 sweep for all pairs and generate summary."""
    logger.info("\n" + "="*80)
    logger.info("SR3 CALENDAR CARRY PHASE-0 VARIANT SWEEP")
    logger.info("="*80)
    logger.info(f"Start date: {start_date}")
    logger.info(f"End date: {end_date}")
    logger.info(f"Pairs to test: {[f'R{lr}-R{sr}' for lr, sr in pairs]}")
    
    summaries = []
    
    for pair in pairs:
        try:
            summary = run_single_pair(
                market=market,
                pair=pair,
                start_date=start_date,
                end_date=end_date,
                break_date=break_date,
                output_base=None
            )
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Error running pair {pair}: {e}")
            continue
    
    if not summaries:
        raise ValueError("No pairs completed successfully")
    
    # Create summary DataFrame
    df = pd.DataFrame(summaries)
    df = df.sort_values('sharpe', ascending=False)
    
    # Save summary
    sleeve_dirs = get_sleeve_dirs(META_SLEEVE, ATOMIC_SLEEVE_ADJACENT)
    summary_dir = sleeve_dirs["base"] / "summary" / "latest"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = summary_dir / "variant_summary.csv"
    df.to_csv(summary_file, index=False)
    logger.info(f"\nSaved variant summary to: {summary_file}")
    
    return df


def select_canonical(df: pd.DataFrame, min_bars: int = 1200) -> dict:
    """
    Select canonical variant based on highest Sharpe with minimum bars constraint.
    
    Special rule: R2-R1 remains canonical unless R1-R0 is clearly superior (not a level proxy).
    R1-R0 is considered "clearly superior" if its Sharpe is at least 0.1 higher than R2-R1.
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
    
    # Check for R2-R1 and R1-R0
    r2r1 = valid[valid['pair'] == 'R2-R1']
    r1r0 = valid[valid['pair'] == 'R1-R0']
    
    # Default to R2-R1 if it exists and meets requirements
    if not r2r1.empty:
        r2r1_sharpe = r2r1.iloc[0]['sharpe']
        # Check if R1-R0 is clearly superior (at least 0.1 Sharpe higher)
        if not r1r0.empty:
            r1r0_sharpe = r1r0.iloc[0]['sharpe']
            if r1r0_sharpe >= r2r1_sharpe + 0.1:
                logger.info(f"R1-R0 is clearly superior (Sharpe {r1r0_sharpe:.4f} vs R2-R1 {r2r1_sharpe:.4f}), selecting R1-R0")
                canonical = r1r0.iloc[0].to_dict()
                return canonical
            else:
                logger.info(f"R2-R1 remains canonical (Sharpe {r2r1_sharpe:.4f} vs R1-R0 {r1r0_sharpe:.4f})")
                canonical = r2r1.iloc[0].to_dict()
                return canonical
        else:
            # R2-R1 exists, R1-R0 doesn't - use R2-R1
            canonical = r2r1.iloc[0].to_dict()
            return canonical
    
    # No R2-R1 found, select highest Sharpe
    valid_sorted = valid.sort_values('sharpe', ascending=False)
    canonical = valid_sorted.iloc[0].to_dict()
    logger.info(f"No R2-R1 found, selecting highest Sharpe: {canonical['pair']} (Sharpe {canonical['sharpe']:.4f})")
    
    return canonical


def main():
    parser = argparse.ArgumentParser(
        description="Run SR3 Calendar Carry Sign-Only Sanity Check (Phase-0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single pair
  python scripts/run_sr3_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --pair 2-1
  
  # Sweep all pairs and select canonical
  python scripts/run_sr3_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --sweep
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
        help="Output directory (default: reports/sanity_checks/carry/sr3_calendar_carry_adjacent/{pair}/archive/<timestamp>)"
    )
    parser.add_argument(
        "--break_date",
        type=str,
        default="2022-01-01",
        help="Date to split subperiods for analysis (default: 2022-01-01)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="spread",
        choices=["option_a", "option_b", "spread"],
        help="Phase-0 variant (only used if not --sweep and not --pair). Default: spread"
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Adjacent rank pair to test (e.g., '2-1', '3-2', '4-3', '5-4'). Only used with variant=spread."
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run variant sweep for all adjacent pairs (2-1, 3-2, 4-3, optionally 5-4) and generate summary"
    )
    parser.add_argument(
        "--min_bars",
        type=int,
        default=1200,
        help="Minimum number of bars required for canonical selection (default: 1200)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize MarketData
        logger.info("\n[1/6] Initializing MarketData broker...")
        market = MarketData()
        logger.info(f"  MarketData universe: {len(market.universe)} symbols")
        
        if args.sweep:
            # Sweep mode: run all pairs
            pairs_to_test = ADJACENT_PAIRS.copy()
            
            # Run sweep
            df_summary = run_sweep(
                market=market,
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
            # Select columns for display, including correlations if available
            display_cols = ['pair', 'sharpe', 'cagr', 'maxdd', 'hitrate', 'vol', 'n_days']
            if 'corr_spread_rank0' in df_summary.columns:
                display_cols.extend(['corr_spread_rank0', 'corr_spread_rank1'])
            print(df_summary[display_cols].to_string(index=False))
            
            print("\n" + "="*80)
            print(f"[CANONICAL] Selected pair: {canonical['pair']}")
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
                ATOMIC_SLEEVE_ADJACENT,
                "phase0",
                path=canonical['latest_path']
            )
            logger.info(f"\nUpdated phase_index/{META_SLEEVE}/{ATOMIC_SLEEVE_ADJACENT}/phase0.txt -> {canonical['latest_path']}")
            
            # Save canonical selection to meta.json in summary directory
            summary_dir = get_sleeve_dirs(META_SLEEVE, ATOMIC_SLEEVE_ADJACENT)["base"] / "summary" / "latest"
            import json
            canonical_meta = {
                'canonical_pair': canonical['pair'],
                'canonical_pair_dir': canonical['pair_dir'],
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
            # Single pair mode
            pair = parse_pair(args.pair)
            summary = run_single_pair(
                market=market,
                pair=pair,
                start_date=args.start,
                end_date=args.end,
                break_date=args.break_date,
                output_base=None
            )
            
            # Update phase index for this pair
            sleeve_dirs = get_sleeve_dirs(META_SLEEVE, ATOMIC_SLEEVE_ADJACENT)
            latest_dir = sleeve_dirs["latest"] / summary['pair_dir']
            update_phase_index(
                META_SLEEVE,
                ATOMIC_SLEEVE_ADJACENT,
                "phase0",
                path=f"sanity_checks/{META_SLEEVE}/{ATOMIC_SLEEVE_ADJACENT}/{summary['pair_dir']}/latest"
            )
            
        else:
            # Legacy mode: run with variant (no pair specified)
            logger.info("=" * 80)
            logger.info("SR3 CALENDAR CARRY SIGN-ONLY SANITY CHECK (Phase-0)")
            logger.info("=" * 80)
            logger.info(f"Start date: {args.start}")
            logger.info(f"End date: {args.end}")
            logger.info(f"Subperiod break: {args.break_date}")
            logger.info(f"Ranks used: 1-4 (canonical Phase-0)")
            if args.variant == "option_a":
                logger.info(f"Variant: Option A - sign(RANK_2 - RANK_1) in rate space")
                logger.info(f"Tradeable: SR3_FRONT_CALENDAR (rank 0)")
            elif args.variant == "option_b":
                logger.info(f"Variant: Option B - sign(mean(RANK_3, RANK_4) - mean(RANK_1, RANK_2)) in rate space")
                logger.info(f"Tradeable: SR3_FRONT_CALENDAR (rank 0)")
            elif args.variant == "spread":
                logger.info(f"Variant: Phase-0C - Trade spread directly (RANK_2 - RANK_1)")
                logger.info(f"Tradeable: Calendar spread (RANK_2 - RANK_1) P&L")
            
            # Run sign-only SR3 carry strategy
            logger.info("\n[2/6] Running sign-only SR3 calendar carry strategy...")
            result = run_sign_only_sr3_carry(
                market=market,
                start_date=args.start,
                end_date=args.end,
                variant=args.variant
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
            if args.variant == "option_a":
                atomic_sleeve_name = ATOMIC_SLEEVE
            else:
                atomic_sleeve_name = f"{ATOMIC_SLEEVE}_{args.variant}"
            sleeve_dirs = get_sleeve_dirs(META_SLEEVE, atomic_sleeve_name)
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
                end_date=args.end,
                variant=args.variant
            )
            
            # Generate plots
            logger.info("[5/6] Generating plots...")
            generate_plots(results=result, output_dir=timestamp_dir)
            
            # Update canonical latest/ directory
            logger.info("\n[6/6] Updating canonical latest/ directory...")
            copy_to_latest(timestamp_dir, latest_dir)
            logger.info(f"  Canonical Phase-0 results: {latest_dir}")
            
            # Print results
            portfolio_sharpe = result['metrics']['sharpe']
            portfolio_cagr = result['metrics']['cagr']
            portfolio_vol = result['metrics']['vol']
            portfolio_maxdd = result['metrics']['maxdd']
            portfolio_hitrate = result['metrics']['hit_rate']
            n_days = result['metrics']['n_days']
            years = result['metrics']['years']
            
            print("\n" + "=" * 80)
            variant_label = "Option A" if args.variant == "option_a" else ("Option B" if args.variant == "option_b" else "Phase-0C (Spread Direct)")
            print(f"SR3 CALENDAR CARRY SIGN-ONLY SANITY CHECK RESULTS (Phase-0, {variant_label})")
            print("=" * 80)
            
            print(f"\nPortfolio:")
            print(f"  CAGR:         {portfolio_cagr:8.2%}")
            print(f"  Vol:          {portfolio_vol:8.2%}")
            print(f"  Sharpe:       {portfolio_sharpe:8.4f}")
            print(f"  MaxDD:        {portfolio_maxdd:8.2%}")
            print(f"  HitRate:      {portfolio_hitrate:8.2%}")
            print(f"  n_days:       {n_days:8d}")
            print(f"  years:        {years:8.2f}")
            
            # Update phase index
            if args.variant == "option_a":
                update_phase_index(META_SLEEVE, ATOMIC_SLEEVE, "phase0")
                logger.info(f"  Updated phase_index/{META_SLEEVE}/{ATOMIC_SLEEVE}/phase0.txt")
            else:
                update_phase_index(META_SLEEVE, atomic_sleeve_name, "phase0")
                logger.info(f"  Updated phase_index/{META_SLEEVE}/{atomic_sleeve_name}/phase0.txt")
        
        logger.info("\n" + "=" * 80)
        logger.info("Diagnostics complete!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'market' in locals():
            market.close()


if __name__ == "__main__":
    main()
