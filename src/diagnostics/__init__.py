"""
Diagnostics module for validating features, sleeves, and overlays.
"""

from .feature_validation import (
    validate_alignment,
    validate_no_lookahead,
    validate_feature_stats,
    run_feature_validation
)

from .sleeve_validation import (
    validate_sleeve,
    backtest_sleeve_only,
    sleeve_correlation_table,
    run_sleeve_validation
)

from .overlay_validation import (
    validate_macro_overlay,
    validate_vol_overlay,
    validate_allocator,
    run_overlay_validation
)

from .combined_validation import (
    validate_combined_strategy,
    run_combined_validation
)

from .minimal_charts import (
    plot_rolling_sharpe,
    plot_weight_heatmap,
    run_minimal_charts
)

from .continuous_price_validation import (
    validate_continuous_prices,
    run_continuous_validation
)

from .universe_consistency import (
    print_universe_stage,
    universe_consistency_report
)

__all__ = [
    # Feature validation
    'validate_alignment',
    'validate_no_lookahead',
    'validate_feature_stats',
    'run_feature_validation',
    # Sleeve validation
    'validate_sleeve',
    'backtest_sleeve_only',
    'sleeve_correlation_table',
    'run_sleeve_validation',
    # Overlay validation
    'validate_macro_overlay',
    'validate_vol_overlay',
    'validate_allocator',
    'run_overlay_validation',
    # Combined validation
    'validate_combined_strategy',
    'run_combined_validation',
    # Minimal charts
    'plot_rolling_sharpe',
    'plot_weight_heatmap',
    'run_minimal_charts',
    # Continuous price validation
    'validate_continuous_prices',
    'run_continuous_validation',
    # Universe consistency
    'print_universe_stage',
    'universe_consistency_report',
]

