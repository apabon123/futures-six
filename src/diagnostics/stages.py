"""
Canonical Stage Definitions for Phase 3B

This module defines the authoritative stage ordering for the Futures-Six execution stack.
All artifact names and diagnostic labels must reference these canonical stage names.

Source of Truth: docs/SOTs/SYSTEM_CONSTRUCTION.md § "Canonical System Architecture"
"""

from enum import Enum
from typing import Dict, List

class Stage(Enum):
    """
    Canonical execution stages in authoritative order.
    
    Stages must be executed in this exact order:
    1. Engine Signals
    2. Engine Policy
    3. Portfolio Construction
    4. Discretionary Overlay (optional)
    5. Risk Targeting
    6. Allocator
    7. Margin & Execution Constraints
    """
    RAW = "raw"  # Pre-policy (engine signals only)
    POST_POLICY = "post_policy"  # After engine policy gating/throttles
    POST_CONSTRUCTION = "post_construction"  # After static portfolio construction aggregation
    POST_DISCRETION = "post_discretion"  # After discretionary overlay (if enabled)
    POST_RISK_TARGETING = "post_risk_targeting"  # After portfolio vol targeting / leverage scaling
    POST_ALLOCATOR = "post_allocator"  # After allocator scalar applied
    TRADED = "traded"  # Final weights used for portfolio return computation (matches post_allocator if allocator applied with proper lag)


# Canonical artifact file names for each stage
STAGE_WEIGHTS_ARTIFACTS: Dict[Stage, str] = {
    Stage.POST_POLICY: "weights_post_policy.csv",
    Stage.POST_CONSTRUCTION: "weights_post_construction.csv",
    Stage.POST_DISCRETION: "weights_post_discretion.csv",
    Stage.POST_RISK_TARGETING: "weights_post_risk_targeting.csv",
    Stage.POST_ALLOCATOR: "weights_post_allocator.csv",
    Stage.TRADED: "weights_used_for_portfolio_returns.csv",  # Final weights used in return computation
}

# Stage display names for diagnostics
STAGE_DISPLAY_NAMES: Dict[Stage, str] = {
    Stage.RAW: "Pre-Construction Aggregate (debugging only)",
    Stage.POST_POLICY: "Post-Policy",
    Stage.POST_CONSTRUCTION: "Post-Construction",
    Stage.POST_DISCRETION: "Post-Discretion",
    Stage.POST_RISK_TARGETING: "Post-RT (blue)",
    Stage.POST_ALLOCATOR: "Post-Allocator",
    Stage.TRADED: "Post-Allocator (traded)",
}

# Ordered list of stages for waterfall attribution
WATERFALL_STAGES: List[Stage] = [
    Stage.RAW,
    Stage.POST_POLICY,
    Stage.POST_RISK_TARGETING,  # Blue stage
    Stage.POST_ALLOCATOR,  # If allocator applied
    Stage.TRADED,  # Final
]

def get_stage_display_name(stage: Stage) -> str:
    """Get human-readable display name for a stage."""
    return STAGE_DISPLAY_NAMES.get(stage, stage.value.replace("_", " ").title())

def get_stage_weights_artifact(stage: Stage) -> str:
    """Get canonical artifact filename for stage weights."""
    return STAGE_WEIGHTS_ARTIFACTS.get(stage, f"weights_{stage.value}.csv")
