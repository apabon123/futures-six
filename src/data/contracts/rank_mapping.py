"""
Contract Rank Mapping Utilities

Provides functions to parse contract series names and extract their rank information.
This ensures correct mapping of contracts to ranks regardless of alphabetical ordering.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def parse_sr3_calendar_rank(series_name: str) -> Optional[int]:
    """
    Parse SR3 calendar contract series name to extract rank.
    
    Rules:
    - Series containing "_FRONT_" or "_FRONT_CALENDAR" → rank 0
    - Series containing "_RANK_k_" where k is a number → rank k
    - Examples:
      - "SR3_FRONT_CALENDAR" → 0
      - "SR3_RANK_1_CALENDAR" → 1
      - "SR3_RANK_2_CALENDAR" → 2
    
    Args:
        series_name: Contract series name (e.g., "SR3_FRONT_CALENDAR", "SR3_RANK_1_CALENDAR")
        
    Returns:
        Integer rank (0, 1, 2, ...) or None if parsing fails
    """
    if not series_name or not isinstance(series_name, str):
        return None
    
    series_upper = series_name.upper()
    
    # Check for FRONT (rank 0)
    if "_FRONT_" in series_upper or series_upper.endswith("_FRONT"):
        return 0
    
    # Check for RANK_k pattern
    # Match patterns like: _RANK_1_, _RANK_2_, etc.
    rank_pattern = r"_RANK[_\s]*(\d+)"
    match = re.search(rank_pattern, series_upper)
    if match:
        try:
            rank = int(match.group(1))
            return rank
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse rank number from '{series_name}': {match.group(1)}")
            return None
    
    # No match found
    return None


def map_contracts_to_ranks(
    series_names: list[str],
    root: str = "SR3",
    parser_func: Optional[callable] = None
) -> dict[str, int]:
    """
    Map contract series names to numeric ranks using parsing.
    
    Args:
        series_names: List of contract series names
        root: Root symbol (e.g., "SR3") - used for validation
        parser_func: Optional custom parser function (default: parse_sr3_calendar_rank)
        
    Returns:
        Dictionary mapping series_name -> rank
        
    Raises:
        ValueError: If parsing fails for any series or duplicates occur
    """
    if parser_func is None:
        parser_func = parse_sr3_calendar_rank
    
    # Parse all series
    series_to_rank = {}
    failed_series = []
    
    for series in series_names:
        rank = parser_func(series)
        if rank is None:
            failed_series.append(series)
        else:
            series_to_rank[series] = rank
    
    # Check for failures
    if failed_series:
        raise ValueError(
            f"Failed to parse ranks for {len(failed_series)} series: {failed_series}. "
            f"Expected format: {root}_FRONT_CALENDAR (rank 0) or {root}_RANK_k_CALENDAR (rank k)"
        )
    
    # Check for duplicate ranks
    rank_to_series = {}
    for series, rank in series_to_rank.items():
        if rank in rank_to_series:
            raise ValueError(
                f"Duplicate rank {rank} found: '{rank_to_series[rank]}' and '{series}'. "
                f"Each rank must map to exactly one series."
            )
        rank_to_series[rank] = series
    
    return series_to_rank

