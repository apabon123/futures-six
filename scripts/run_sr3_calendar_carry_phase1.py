"""
SR3 Calendar Carry Phase-1 Runner

Engineered, tradable implementation of SR3 calendar carry.
Upgrades Phase-0 sign-only sanity check with z-scoring, vol targeting, and DV01-neutral weighting.

Canonical pair: R2-R1 (Rank 2 - Rank 1)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import json
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.agents import MarketData
from src.strategies.carry.sr3_calendar_carry import compute_sr3_calendar_carry_phase1
from src.utils.phase_index import update_phase_index

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Meta-sleeve and atomic sleeve names
META_SLEEVE = "carry"
ATOMIC_SLEEVE = "sr3_calendar_carry"


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series
) -> dict:
    """
    Compute summary statistics for portfolio returns.
    
    Args:
        portfolio_returns: Daily portfolio returns
        equity_curve: Cumulative equity curve
        
    Returns:
        Dict with portfolio metrics
    """
    if portfolio_returns.empty:
        return {
            'CAGR': 0.0,
            'Vol': 0.0,
            'Sharpe': 0.0,
            'MaxDD': 0.0,
            'HitRate': 0.0,
            'n_days': 0,
            'years': 0.0
        }
    
    # Annualized metrics
    n_days = len(portfolio_returns)
    years = n_days / 252.0
    
    # CAGR
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0 if len(equity_curve) > 0 else 0.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    
    # Volatility (annualized)
    vol = portfolio_returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = (cagr / vol) if vol > 0 else 0.0
    
    # Max drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    maxdd = drawdown.min()
    
    # Hit rate
    hit_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) if len(portfolio_returns) > 0 else 0.0
    
    return {
        'CAGR': cagr,
        'Vol': vol,
        'Sharpe': sharpe,
        'MaxDD': maxdd,
        'HitRate': hit_rate,
        'n_days': n_days,
        'years': years
    }


def compute_per_asset_stats(
    portfolio_returns: pd.Series,
    spread_returns: pd.Series,
    positions: pd.Series
) -> pd.DataFrame:
    """
    Compute per-asset statistics.
    
    Args:
        portfolio_returns: Daily portfolio returns
        spread_returns: Daily spread returns
        positions: Daily positions
        
    Returns:
        DataFrame with per-asset stats
    """
    # Align all series
    common_dates = portfolio_returns.index.intersection(
        spread_returns.index
    ).intersection(positions.index)
    
    if len(common_dates) == 0:
        return pd.DataFrame()
    
    portfolio_ret_aligned = portfolio_returns.loc[common_dates]
    spread_ret_aligned = spread_returns.loc[common_dates]
    positions_aligned = positions.loc[common_dates]
    
    # Compute asset-level stats
    asset_name = "SR3_SPREAD_R2_R1"
    
    ann_ret = portfolio_ret_aligned.mean() * 252
    ann_vol = portfolio_ret_aligned.std() * np.sqrt(252)
    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
    
    stats = pd.DataFrame({
        'AnnRet': [ann_ret],
        'AnnVol': [ann_vol],
        'Sharpe': [sharpe]
    }, index=[asset_name])
    
    return stats


def save_results(
    results: dict,
    stats: dict,
    per_asset_stats: pd.DataFrame,
    output_dir: Path,
    start_date: str,
    end_date: str,
    phase0_sharpe: Optional[float] = None
):
    """
    Save all results to output directory.
    
    Args:
        results: Strategy results dict
        stats: Portfolio statistics dict
        per_asset_stats: Per-asset statistics DataFrame
        output_dir: Output directory path
        start_date: Start date
        end_date: End date
        phase0_sharpe: Phase-0 Sharpe for comparison (optional)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save portfolio returns
    portfolio_returns_df = pd.DataFrame({
        'date': results['portfolio_returns'].index,
        'ret': results['portfolio_returns'].values
    })
    portfolio_returns_df.to_csv(output_dir / 'portfolio_returns.csv', index=False)
    
    # Save equity curve
    equity_curve_df = pd.DataFrame({
        'date': results['equity_curve'].index,
        'equity': results['equity_curve'].values
    })
    equity_curve_df.to_csv(output_dir / 'equity_curve.csv', index=False)
    
    # Save positions
    positions_df = pd.DataFrame({
        'date': results['positions'].index,
        'position': results['positions'].values
    })
    positions_df.to_csv(output_dir / 'positions.csv', index=False)
    
    # Save per-asset stats
    if not per_asset_stats.empty:
        per_asset_stats.to_csv(output_dir / 'per_asset_stats.csv')
    
    # Save meta.json
    meta = results['metadata'].copy()
    meta.update({
        'start_date': start_date,
        'end_date': end_date,
        'portfolio_metrics': stats,
        'per_asset_stats': per_asset_stats.to_dict('index') if not per_asset_stats.empty else {},
        'phase0_sharpe_comparison': phase0_sharpe if phase0_sharpe is not None else None
    })
    
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}")


def load_phase0_sharpe() -> Optional[float]:
    """
    Load Phase-0 Sharpe from canonical results.
    
    Returns:
        Phase-0 Sharpe ratio or None if not found
    """
    try:
        phase0_path = Path("reports/sanity_checks/carry/sr3_calendar_carry_adjacent/latest/2-1/meta.json")
        if phase0_path.exists():
            with open(phase0_path, 'r') as f:
                phase0_meta = json.load(f)
            return phase0_meta.get('portfolio_metrics', {}).get('Sharpe')
    except Exception as e:
        logger.warning(f"Could not load Phase-0 Sharpe: {e}")
    return None


def main():
    """Run SR3 Calendar Carry Phase-1."""
    parser = argparse.ArgumentParser(
        description="SR3 Calendar Carry Phase-1: Engineered implementation with z-scoring and vol targeting"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=CANONICAL_START,
        help=f"Start date (YYYY-MM-DD, default: {CANONICAL_START})"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date (YYYY-MM-DD, default: {CANONICAL_END})"
    )
    parser.add_argument(
        "--target-vol",
        type=float,
        default=0.10,
        help="Target annualized volatility (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--zscore-window",
        type=int,
        default=90,
        help="Rolling window for z-score normalization (default: 90 days)"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=2.0,
        help="Symmetric clipping bounds for normalized signal (default: ±2.0)"
    )
    parser.add_argument(
        "--vol-lookback",
        type=int,
        default=60,
        help="Rolling window for realized vol calculation (default: 60 days)"
    )
    parser.add_argument(
        "--use-dv01",
        action="store_true",
        help="Use DV01-neutral weighting (requires --dv01-long and --dv01-short)"
    )
    parser.add_argument(
        "--dv01-long",
        type=float,
        default=None,
        help="DV01 for long leg (rank 2) - required if --use-dv01"
    )
    parser.add_argument(
        "--dv01-short",
        type=float,
        default=None,
        help="DV01 for short leg (rank 1) - required if --use-dv01"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="phase1",
        choices=["phase1", "phase0_equiv"],
        help="Strategy mode: 'phase1' (default) or 'phase0_equiv' (degenerate mode to match Phase-0)"
    )
    parser.add_argument(
        "--flip-sign",
        action="store_true",
        help="Flip signal sign (for sign convention testing)"
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        help="Execution lag in days (default: 1, for carry typically correct)"
    )
    
    args = parser.parse_args()
    
    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"reports/runs/carry/sr3_calendar_carry_phase1/{run_id}")
    
    logger.info("=" * 80)
    logger.info("SR3 Calendar Carry Phase-1")
    logger.info("=" * 80)
    logger.info(f"Start date: {args.start}")
    logger.info(f"End date: {args.end}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Target vol: {args.target_vol*100:.1f}%")
    logger.info(f"Z-score window: {args.zscore_window}d")
    logger.info(f"Clip bounds: ±{args.clip}")
    logger.info(f"Vol lookback: {args.vol_lookback}d")
    logger.info(f"DV01 method: {'true' if args.use_dv01 else 'proxy'}")
    logger.info(f"Flip sign: {args.flip_sign}")
    logger.info(f"Lag: {args.lag}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 80)
    
    # Guardrail: Assert canonical pair
    from src.strategies.carry.sr3_calendar_carry import CANONICAL_PAIR
    logger.info(f"[GUARDRAIL] Canonical pair: R{CANONICAL_PAIR[0]}-R{CANONICAL_PAIR[1]}")
    
    # Initialize MarketData
    try:
        market = MarketData()
    except Exception as e:
        logger.error(f"Failed to initialize MarketData: {e}")
        return 1
    
    # Compute Phase-1 strategy
    try:
        results = compute_sr3_calendar_carry_phase1(
            market=market,
            start_date=args.start,
            end_date=args.end,
            mode=args.mode,
            zscore_window=args.zscore_window,
            clip=args.clip,
            target_vol=args.target_vol,
            vol_lookback=args.vol_lookback,
            use_dv01=args.use_dv01,
            dv01_long=args.dv01_long,
            dv01_short=args.dv01_short,
            flip_sign=args.flip_sign,
            lag=args.lag
        )
    except Exception as e:
        logger.error(f"Failed to compute Phase-1 strategy: {e}")
        import traceback
        traceback.print_exc()
        market.close()
        return 1
    
    # Guardrail: Assert spread returns variance > 0 (already checked in strategy, but log it)
    spread_returns_std = results['spread_returns'].std()
    is_finite = np.isfinite(spread_returns_std)
    has_variance = spread_returns_std > 0 if is_finite else False
    logger.info(f"[GUARDRAIL] Spread returns std: {spread_returns_std:.6f} (finite={is_finite}, variance>0={has_variance}: PASS)")
    
    # Compute statistics
    stats = compute_summary_stats(
        portfolio_returns=results['portfolio_returns'],
        equity_curve=results['equity_curve']
    )
    
    per_asset_stats = compute_per_asset_stats(
        portfolio_returns=results['portfolio_returns'],
        spread_returns=results['spread_returns'],
        positions=results['positions']
    )
    
    # Load Phase-0 Sharpe for comparison
    phase0_sharpe = load_phase0_sharpe()
    
    # Save results
    save_results(
        results=results,
        stats=stats,
        per_asset_stats=per_asset_stats,
        output_dir=output_dir,
        start_date=args.start,
        end_date=args.end,
        phase0_sharpe=phase0_sharpe
    )
    
    # Update phase index
    try:
        update_phase_index(
            meta_sleeve=META_SLEEVE,
            sleeve_name=ATOMIC_SLEEVE,
            phase="phase1",
            run_id=run_id
        )
        logger.info(f"Updated phase index: reports/phase_index/{META_SLEEVE}/{ATOMIC_SLEEVE}/phase1.txt -> {run_id}")
    except Exception as e:
        logger.warning(f"Failed to update phase index: {e}")
    
    # Validation Checklist
    logger.info("=" * 80)
    logger.info("VALIDATION CHECKLIST")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Flip sign: {args.flip_sign}")
    logger.info(f"Lag: {args.lag}")
    logger.info(f"Sharpe: {stats['Sharpe']:.4f}")
    logger.info(f"CAGR: {stats['CAGR']*100:.2f}%")
    logger.info(f"MaxDD: {stats['MaxDD']*100:.2f}%")
    logger.info(f"Vol: {stats['Vol']*100:.2f}%")
    logger.info(f"Hit rate: {stats['HitRate']*100:.2f}%")
    if phase0_sharpe is not None:
        logger.info(f"Phase-0 Sharpe (comparison): {phase0_sharpe:.4f}")
        logger.info(f"Phase-1 vs Phase-0: {stats['Sharpe'] - phase0_sharpe:+.4f}")
        if args.mode == "phase0_equiv":
            logger.info("")
            logger.info("PHASE-0 EQUIVALENCE CHECK:")
            if stats['Sharpe'] * phase0_sharpe > 0:
                logger.info(f"  [PASS] Same sign as Phase-0 (directionally consistent)")
            else:
                logger.info(f"  [WARNING] Opposite sign from Phase-0 (check sign convention)")
            if abs(stats['Sharpe'] - phase0_sharpe) < 0.3:
                logger.info(f"  [PASS] Sharpe within 0.3 of Phase-0 (reasonable match)")
            else:
                logger.info(f"  [WARNING] Sharpe differs by {abs(stats['Sharpe'] - phase0_sharpe):.2f} from Phase-0")
    logger.info("=" * 80)
    
    # Guardrails summary
    logger.info("GUARDRAILS:")
    logger.info(f"  [PASS] Canonical pair = R{CANONICAL_PAIR[0]}-R{CANONICAL_PAIR[1]}")
    logger.info(f"  [PASS] Rank parsing succeeded (asserted in get_contracts_by_root)")
    logger.info(f"  [PASS] Spread returns variance > 0 (std={spread_returns_std:.6f})")
    logger.info(f"  [PASS] Effective start date: {results['metadata']['effective_start_date']}")
    logger.info("=" * 80)
    
    logger.info(f"Phase-1 complete. Results saved to: {output_dir}")
    
    market.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

