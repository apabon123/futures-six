"""
ParamSweepRunner: Grid/Latin Hypercube Parameter Sweep Agent

Orchestrates parameter sweeps over strategy configurations to find optimal settings.
Supports:
- Grid search (exhaustive combinations)
- Latin hypercube sampling (efficient sampling for high-dimensional spaces)
- Parallelized execution with process pools
- Deterministic results given same seed
- Tidy output tables with performance metrics

No data writes during execution. All results written to /reports/sweeps/<timestamp>/.
"""

import logging
import copy
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from multiprocessing import Pool, cpu_count
import warnings

import pandas as pd
import numpy as np
import yaml

# Suppress warnings in parallel processes
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def _set_nested_value(config: dict, key_path: str, value: Any) -> dict:
    """
    Set a nested dictionary value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "macro.vol_thresholds.low")
        value: Value to set
        
    Returns:
        Modified config dictionary
        
    Example:
        _set_nested_value({}, "macro.vol_thresholds.low", 0.12)
        -> {"macro": {"vol_thresholds": {"low": 0.12}}}
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate/create nested structure
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set final value
    current[keys[-1]] = value
    
    return config


def _apply_params_to_config(base_config: dict, params: dict) -> dict:
    """
    Apply parameter set to base configuration.
    
    Args:
        base_config: Base configuration dictionary
        params: Parameter dictionary with dot-notation keys
        
    Returns:
        New configuration with parameters applied
    """
    config = copy.deepcopy(base_config)
    
    for key, value in params.items():
        _set_nested_value(config, key, value)
    
    return config


def _build_components(config: dict, market):
    """
    Build strategy components from configuration.
    
    Args:
        config: Configuration dictionary
        market: MarketData instance
        
    Returns:
        Dict of initialized components
    """
    from src.agents.strat_momentum import TSMOM
    from src.agents.overlay_volmanaged import VolManagedOverlay
    from src.agents.overlay_macro_regime import MacroRegimeFilter
    from src.agents.risk_vol import RiskVol
    from src.agents.allocator import Allocator
    
    # Extract configs
    tsmom_cfg = config.get('tsmom', {})
    vol_overlay_cfg = config.get('vol_overlay', {})
    macro_cfg = config.get('macro_regime', {})
    risk_cfg = config.get('risk_vol', {})
    alloc_cfg = config.get('allocator', {})
    
    # Initialize components
    strategy = TSMOM(
        lookbacks=tsmom_cfg.get('lookbacks', [252]),
        skip_recent=tsmom_cfg.get('skip_recent', 21),
        standardize=tsmom_cfg.get('standardize', 'vol'),
        signal_cap=tsmom_cfg.get('signal_cap', 3.0),
        rebalance=tsmom_cfg.get('rebalance', 'W-FRI'),
    )
    
    risk_vol = RiskVol(
        cov_lookback=risk_cfg.get('cov_lookback', 252),
        vol_lookback=risk_cfg.get('vol_lookback', 63),
        shrinkage=risk_cfg.get('shrinkage', 'lw'),
        nan_policy=risk_cfg.get('nan_policy', 'mask-asset')
    )
    
    vol_overlay = VolManagedOverlay(
        risk_vol=risk_vol,
        target_vol=vol_overlay_cfg.get('target_vol', 0.20),
        lookback_vol=vol_overlay_cfg.get('lookback_vol', 63),
        leverage_mode=vol_overlay_cfg.get('leverage_mode', 'global'),
        cap_leverage=vol_overlay_cfg.get('cap_leverage', 7.0),
        position_bounds=vol_overlay_cfg.get('position_bounds', [-3.0, 3.0])
    )
    
    # Macro overlay is optional
    macro_overlay = None
    if macro_cfg and macro_cfg.get('vol_thresholds'):
        macro_overlay = MacroRegimeFilter(
            rebalance=macro_cfg.get('rebalance', 'W-FRI'),
            vol_thresholds=macro_cfg.get('vol_thresholds'),
            k_bounds=macro_cfg.get('k_bounds', {'min': 0.5, 'max': 1.0}),
            smoothing=macro_cfg.get('smoothing', 0.15),
            vol_lookback=macro_cfg.get('vol_lookback', 21),
            breadth_lookback=macro_cfg.get('breadth_lookback', 200),
            proxy_symbols=tuple(macro_cfg.get('proxy_symbols', ['ES_FRONT_CALENDAR_2D', 'NQ_FRONT_CALENDAR_2D']))
        )
    
    allocator = Allocator(
        method=alloc_cfg.get('method', 'signal-beta'),
        gross_cap=alloc_cfg.get('gross_cap', 7.0),
        net_cap=alloc_cfg.get('net_cap', 2.0),
        w_bounds_per_asset=alloc_cfg.get('w_bounds_per_asset', [-1.5, 1.5]),
        turnover_cap=alloc_cfg.get('turnover_cap', 0.5),
        lambda_turnover=alloc_cfg.get('lambda_turnover', 0.001)
    )
    
    return {
        'strategy': strategy,
        'overlay': vol_overlay,
        'macro_overlay': macro_overlay,
        'risk_vol': risk_vol,
        'allocator': allocator
    }


def _run_single_backtest(args):
    """
    Run a single backtest for one parameter combination.
    
    This function is designed to be called by multiprocessing.Pool.
    
    Args:
        args: Tuple of (param_dict, base_config, start, end, seed)
        
    Returns:
        Dict with params and metrics
    """
    params, base_config, start, end, seed = args
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    try:
        # Import here to avoid issues with multiprocessing
        from src.agents import MarketData
        from src.agents.exec_sim import ExecSim
        
        # Apply parameters to config
        config = _apply_params_to_config(base_config, params)
        
        # Initialize MarketData
        market = MarketData()
        
        # Build components
        components = _build_components(config, market)
        
        # Pre-compute strategy rebalance schedule
        if hasattr(components['strategy'], 'fit_in_sample'):
            components['strategy'].fit_in_sample(market, start=start, end=end)
        
        # Get rebalance frequency from exec config
        exec_cfg = config.get('exec', {})
        rebalance = exec_cfg.get('rebalance', 'W-FRI')
        slippage_bps = exec_cfg.get('slippage_bps', 0.5)
        
        # Initialize ExecSim
        exec_sim = ExecSim(
            rebalance=rebalance,
            slippage_bps=slippage_bps,
            commission_per_contract=exec_cfg.get('commission_per_contract', 0.0),
            cash_rate=exec_cfg.get('cash_rate', 0.0),
            position_notional_scale=exec_cfg.get('position_notional_scale', 1.0)
        )
        
        # Run backtest
        results = exec_sim.run(
            market=market,
            start=start,
            end=end,
            components=components
        )
        
        # Close market connection
        market.close()
        
        # Extract metrics
        report = results['report']
        
        # Calculate Calmar ratio
        calmar = report['cagr'] / abs(report['max_drawdown']) if report['max_drawdown'] != 0 else 0.0
        
        # Calculate cost drag (slippage * avg_turnover)
        cost_drag = (slippage_bps / 10000) * report.get('avg_turnover', 0.0)
        
        # Prepare result row
        result = {**params}  # Include all parameters
        result.update({
            'cagr': report['cagr'],
            'vol': report['vol'],
            'sharpe': report['sharpe'],
            'max_drawdown': report['max_drawdown'],
            'calmar': calmar,
            'turnover': report.get('avg_turnover', 0.0),
            'cost_drag': cost_drag,
            'hit_rate': report.get('hit_rate', 0.0),
            'gross_leverage': report.get('avg_gross', 0.0),
            'net_exposure': report.get('avg_net', 0.0),
            'n_periods': report.get('n_periods', 0),
            'seed': seed,
            'success': True,
            'error': None
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in backtest with params {params}: {e}")
        result = {**params}
        result.update({
            'cagr': np.nan,
            'vol': np.nan,
            'sharpe': np.nan,
            'max_drawdown': np.nan,
            'calmar': np.nan,
            'turnover': np.nan,
            'cost_drag': np.nan,
            'hit_rate': np.nan,
            'gross_leverage': np.nan,
            'net_exposure': np.nan,
            'n_periods': 0,
            'seed': seed,
            'success': False,
            'error': str(e)
        })
        return result


def _generate_grid_combinations(grid: Dict[str, List]) -> List[Dict]:
    """
    Generate all combinations from a parameter grid.
    
    Args:
        grid: Dict mapping parameter names to lists of values
        
    Returns:
        List of parameter dictionaries
        
    Example:
        grid = {"a": [1, 2], "b": [3, 4]}
        -> [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]
    """
    keys = list(grid.keys())
    values = list(grid.values())
    
    combinations = []
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)
    
    return combinations


def run_sweep(
    base_config: dict,
    grid: Dict[str, List],
    seeds: List[int] = None,
    start: str = "2021-01-01",
    end: str = "2025-11-05",
    method: str = "grid",
    n_samples: Optional[int] = None,
    n_workers: Optional[int] = None,
    output_dir: Optional[str] = None,
    save_top_n: int = 10
) -> pd.DataFrame:
    """
    Run parameter sweep over configuration space.
    
    Args:
        base_config: Base configuration dictionary (will be modified by grid params)
        grid: Parameter grid mapping dot-notation keys to value lists
              Example: {"macro.vol_thresholds.low": [0.10, 0.12, 0.14]}
        seeds: List of random seeds for reproducibility (default: [0])
        start: Backtest start date
        end: Backtest end date
        method: Sweep method ("grid" or "latin" for Latin hypercube)
        n_samples: Number of samples for Latin hypercube (ignored for grid)
        n_workers: Number of parallel workers (default: cpu_count() - 1)
        output_dir: Output directory for results (default: reports/sweeps/<timestamp>)
        save_top_n: Number of top configurations to save as YAML files
        
    Returns:
        DataFrame with one row per parameter combination, containing:
        - All parameter values
        - Performance metrics: cagr, vol, sharpe, max_drawdown, calmar, turnover, cost_drag
        - Success flag and error message if failed
        
    Example:
        >>> config = {"tsmom": {"lookbacks": [252]}, "exec": {"rebalance": "W-FRI"}}
        >>> grid = {
        ...     "macro.vol_thresholds.low": [0.10, 0.12],
        ...     "tsmom.lookbacks": [[252], [126, 252]]
        ... }
        >>> results = run_sweep(config, grid)
    """
    if seeds is None:
        seeds = [0]
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    logger.info("=" * 80)
    logger.info("PARAMETER SWEEP RUNNER")
    logger.info("=" * 80)
    
    # Generate parameter combinations
    if method == "grid":
        param_combinations = _generate_grid_combinations(grid)
        logger.info(f"Grid search: {len(param_combinations)} combinations")
    elif method == "latin":
        if n_samples is None:
            raise ValueError("n_samples required for Latin hypercube sampling")
        # TODO: Implement Latin hypercube sampling
        raise NotImplementedError("Latin hypercube sampling not yet implemented")
    else:
        raise ValueError(f"Unknown sweep method: {method}")
    
    logger.info(f"Backtest period: {start} to {end}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Workers: {n_workers}")
    logger.info(f"Total backtests: {len(param_combinations) * len(seeds)}")
    
    # Prepare arguments for parallel execution
    tasks = []
    for params in param_combinations:
        for seed in seeds:
            tasks.append((params, base_config, start, end, seed))
    
    logger.info(f"\nStarting {len(tasks)} backtests...")
    
    # Run backtests in parallel
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_run_single_backtest, tasks)
    else:
        # Serial execution for debugging
        results = [_run_single_backtest(task) for task in tasks]
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Log summary
    n_success = df['success'].sum()
    n_failed = len(df) - n_success
    
    logger.info(f"\nCompleted: {n_success} successful, {n_failed} failed")
    
    if n_success > 0:
        successful = df[df['success']]
        logger.info(f"\nPerformance Summary (successful runs):")
        logger.info(f"  CAGR:   mean={successful['cagr'].mean():.2%}, "
                   f"min={successful['cagr'].min():.2%}, max={successful['cagr'].max():.2%}")
        logger.info(f"  Sharpe: mean={successful['sharpe'].mean():.2f}, "
                   f"min={successful['sharpe'].min():.2f}, max={successful['sharpe'].max():.2f}")
        logger.info(f"  MaxDD:  mean={successful['max_drawdown'].mean():.2%}, "
                   f"min={successful['max_drawdown'].min():.2%}, max={successful['max_drawdown'].max():.2%}")
    
    # Save results
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"reports/sweeps/{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary CSV
    summary_file = output_path / "summary.csv"
    df.to_csv(summary_file, index=False)
    logger.info(f"\nSaved summary to: {summary_file}")
    
    # Save top N configurations (by Sharpe ratio)
    if n_success > 0 and save_top_n > 0:
        successful_df = df[df['success']].copy()
        top_configs = successful_df.nlargest(save_top_n, 'sharpe')
        
        for i, (idx, row) in enumerate(top_configs.iterrows(), 1):
            # Extract parameter columns (exclude metrics)
            metric_cols = ['cagr', 'vol', 'sharpe', 'max_drawdown', 'calmar', 
                          'turnover', 'cost_drag', 'hit_rate', 'gross_leverage', 
                          'net_exposure', 'n_periods', 'seed', 'success', 'error']
            param_cols = [col for col in df.columns if col not in metric_cols]
            
            # Build config from parameters
            top_config = copy.deepcopy(base_config)
            for col in param_cols:
                _set_nested_value(top_config, col, row[col])
            
            # Add metadata
            top_config['_metadata'] = {
                'rank': i,
                'sharpe': float(row['sharpe']),
                'cagr': float(row['cagr']),
                'vol': float(row['vol']),
                'max_drawdown': float(row['max_drawdown']),
                'calmar': float(row['calmar']),
                'seed': int(row['seed'])
            }
            
            # Save to YAML
            config_file = output_path / f"top_{i:02d}_sharpe_{row['sharpe']:.2f}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(top_config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"  Rank {i}: Sharpe={row['sharpe']:.2f}, CAGR={row['cagr']:.2%} -> {config_file.name}")
    
    logger.info("=" * 80)
    
    return df


def compare_configs(
    configs: Dict[str, dict],
    start: str = "2021-01-01",
    end: str = "2025-11-05",
    seed: int = 0,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple named configurations.
    
    Convenience function for running specific configurations (e.g., "Baseline vs Macro vs Macro+XSec").
    
    Args:
        configs: Dict mapping config names to config dictionaries
        start: Backtest start date
        end: Backtest end date
        seed: Random seed
        output_dir: Output directory (default: reports/sweeps/comparison_<timestamp>)
        
    Returns:
        DataFrame with one row per configuration
        
    Example:
        >>> configs = {
        ...     "Baseline": baseline_config,
        ...     "Macro": macro_config,
        ...     "Macro+XSec": macro_xsec_config
        ... }
        >>> results = compare_configs(configs)
    """
    logger.info("=" * 80)
    logger.info("CONFIGURATION COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Comparing {len(configs)} configurations")
    
    results = []
    for name, config in configs.items():
        logger.info(f"\nRunning: {name}")
        
        # Create a dummy grid with just the config name
        task = ({}, config, start, end, seed)
        result = _run_single_backtest(task)
        result['config_name'] = name
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Reorder columns to put config_name first
    cols = ['config_name'] + [col for col in df.columns if col != 'config_name']
    df = df[cols]
    
    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)
    
    for _, row in df.iterrows():
        logger.info(f"\n{row['config_name']}:")
        logger.info(f"  CAGR:        {row['cagr']:8.2%}")
        logger.info(f"  Volatility:  {row['vol']:8.2%}")
        logger.info(f"  Sharpe:      {row['sharpe']:8.2f}")
        logger.info(f"  Max DD:      {row['max_drawdown']:8.2%}")
        logger.info(f"  Calmar:      {row['calmar']:8.2f}")
        logger.info(f"  Turnover:    {row['turnover']:8.2f}")
    
    # Save results
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"reports/sweeps/comparison_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_path / "comparison.csv"
    df.to_csv(summary_file, index=False)
    logger.info(f"\nSaved comparison to: {summary_file}")
    logger.info("=" * 80)
    
    return df

