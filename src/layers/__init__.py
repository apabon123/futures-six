"""
Canonical System Layers

This module contains the canonical execution stack layers as defined in SYSTEM_CONSTRUCTION.md:

1. Engine Signals (alpha) - in src/agents/strat_*.py
2. Engine Policy (gates/throttles) - PLANNED
3. Portfolio Construction (static weights) - in configs/strategies.yaml
4. Discretionary Overlay (bounded tilts) - PLANNED
5. Risk Targeting (vol → leverage) - risk_targeting.py
6. Allocator (risk brake) - in src/allocator/
7. Margin & Execution Constraints - in src/agents/exec_sim.py

Each layer has single responsibility and explicit exclusions.
"""

from .risk_targeting import RiskTargetingLayer, create_risk_targeting_layer
from .artifact_writer import ArtifactWriter, create_artifact_writer

__all__ = [
    "RiskTargetingLayer",
    "create_risk_targeting_layer",
    "ArtifactWriter",
    "create_artifact_writer",
]

