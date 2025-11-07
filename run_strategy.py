"""
Main entry point for running the futures-six TSMOM strategy.

This script orchestrates the full pipeline:
1. MarketData broker - read OHLCV data
2. TSMOM strategy - generate momentum signals
3. MacroRegimeFilter - regime scaler applied to signals
4. RiskVol - calculate volatility and covariance
5. VolManaged overlay - scale signals by volatility target
6. Allocator - convert signals to portfolio weights
7. ExecSim - simulate execution and generate backtest results
"""

import sys
from pathlib import Path
import pandas as pd
import logging
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = Path("configs/strategies.yaml")

# Import agents
from src.agents import MarketData
from src.agents.strat_momentum import TSMOM
from src.agents.overlay_volmanaged import VolManagedOverlay
from src.agents.overlay_macro_regime import MacroRegimeFilter
from src.agents.risk_vol import RiskVol
from src.agents.allocator import Allocator
from src.agents.exec_sim import ExecSim


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load strategy configuration from YAML."""
    if not path.exists():
        logger.warning("Config file not found at %s; using defaults.", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    """Run the complete strategy backtest."""
    
    logger.info("=" * 80)
    logger.info("FUTURES-SIX: TSMOM Strategy Backtest")
    logger.info("=" * 80)
    
    # Configuration
    start_date = "2021-01-01"  # Start after some history for lookbacks
    end_date = "2025-11-05"    # Latest available data
    
    logger.info(f"\nBacktest Period: {start_date} to {end_date}")
    
    try:
        config = load_config()
        macro_cfg = config.get("macro_regime", {}) if config else {}
        
        # 1. Initialize MarketData
        logger.info("\n[1/7] Initializing MarketData broker...")
        market = MarketData()
        logger.info(f"  Universe: {market.universe}")
        logger.info(f"  Table: {market.table_name}")
        
        # 2. Initialize TSMOM Strategy
        logger.info("\n[2/7] Initializing TSMOM strategy...")
        strategy = TSMOM()
        logger.info(f"  Config: {strategy.describe()}")
        
        # Pre-compute rebalance schedule
        strategy.fit_in_sample(market, start=start_date, end=end_date)
        
        # 3. Initialize MacroRegimeFilter
        logger.info("\n[3/7] Initializing MacroRegimeFilter...")
        macro_overlay = MacroRegimeFilter(
            rebalance=macro_cfg.get("rebalance", "W-FRI"),
            vol_thresholds=macro_cfg.get("vol_thresholds"),
            k_bounds=macro_cfg.get("k_bounds"),
            smoothing=macro_cfg.get("smoothing", 0.2),
            vol_lookback=macro_cfg.get("vol_lookback", 21),
            breadth_lookback=macro_cfg.get("breadth_lookback", 200),
            proxy_symbols=tuple(macro_cfg.get("proxy_symbols", ("ES", "NQ")))
        )
        logger.info(f"  Config: {macro_overlay.describe()}")
        
        # 4. Initialize RiskVol (needed by VolManaged)
        logger.info("\n[4/7] Initializing RiskVol agent...")
        risk = RiskVol()
        logger.info(f"  Config: {risk.describe()}")
        
        # 5. Initialize VolManaged Overlay
        logger.info("\n[5/7] Initializing VolManaged overlay...")
        vol_overlay = VolManagedOverlay(risk_vol=risk)
        logger.info(f"  Config: {vol_overlay.describe()}")
        
        # 6. Initialize Allocator
        logger.info("\n[6/7] Initializing Allocator...")
        allocator = Allocator()
        logger.info(f"  Config: {allocator.describe()}")
        
        # 7. Initialize and Run ExecSim
        logger.info("\n[7/7] Running backtest with ExecSim...")
        exec_sim = ExecSim()
        logger.info(f"  Config: {exec_sim.describe()}")
        
        # Run backtest
        logger.info(f"\nExecuting backtest from {start_date} to {end_date}...")
        
        # Package components
        components = {
            'strategy': strategy,
            'macro_overlay': macro_overlay,
            'overlay': vol_overlay,
            'risk_vol': risk,
            'allocator': allocator
        }
        
        results = exec_sim.run(
            market=market,
            start=start_date,
            end=end_date,
            components=components
        )
        
        # Extract results
        equity = results['equity_curve']
        report = results['report']
        weights_panel = results['weights_panel']
        signals_panel = results['signals_panel']
        
        # Display Results
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"\nEquity Curve:")
        logger.info(f"  Start Date:     {equity.index[0]}")
        logger.info(f"  End Date:       {equity.index[-1]}")
        logger.info(f"  Starting Value: ${equity.iloc[0]:,.2f}")
        logger.info(f"  Ending Value:   ${equity.iloc[-1]:,.2f}")
        logger.info(f"  Total Return:   {(equity.iloc[-1] / equity.iloc[0] - 1) * 100:.2f}%")
        
        logger.info(f"\nPerformance Metrics:")
        for metric, value in report.items():
            if isinstance(value, (int, float)):
                if 'vol' in metric.lower() or 'drawdown' in metric.lower() or 'sharpe' in metric.lower() or 'turnover' in metric.lower() or 'rate' in metric.lower():
                    logger.info(f"  {metric:20}: {value:8.4f}")
                elif 'cagr' in metric.lower() or 'return' in metric.lower():
                    logger.info(f"  {metric:20}: {value:8.2%}")
                elif 'gross' in metric.lower() or 'net' in metric.lower():
                    logger.info(f"  {metric:20}: {value:8.2f}x")
                else:
                    logger.info(f"  {metric:20}: {value}")
            else:
                logger.info(f"  {metric:20}: {value}")
        
        logger.info(f"\nPortfolio Statistics:")
        logger.info(f"  Number of rebalances: {len(weights_panel)}")
        logger.info(f"  Assets in universe:   {len(market.universe)}")
        
        # Show final weights
        if not weights_panel.empty:
            final_weights = weights_panel.iloc[-1]
            logger.info(f"\nFinal Portfolio Weights (as of {weights_panel.index[-1]}):")
            for symbol, weight in final_weights.items():
                if abs(weight) > 0.001:  # Only show non-zero weights
                    logger.info(f"  {symbol:25}: {weight:7.2%}")
        
        # Close market data connection
        market.close()
        
        logger.info("\n" + "=" * 80)
        logger.info("Backtest completed successfully!")
        logger.info("=" * 80)
        
        return results
    
    except Exception as e:
        logger.error(f"\nError during backtest: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results is not None else 1)

