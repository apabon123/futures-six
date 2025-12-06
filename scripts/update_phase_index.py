"""
Helper script to update phase index for Phase-1/2 runs.

Usage:
    # Update Phase-1 index
    python scripts/update_phase_index.py trend breakout_mid_50_100 phase1 breakout_1b_7030
    
    # Update Phase-2 index
    python scripts/update_phase_index.py trend breakout_mid_50_100 phase2 core_v3_tsb_phase2
    
    # Set sleeve status
    python scripts/update_phase_index.py trend persistence status "PARKED after Phase-1 fail"
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.phase_index import update_phase_index, set_sleeve_status


def main():
    parser = argparse.ArgumentParser(
        description="Update phase index for a sleeve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update Phase-1 index
  python scripts/update_phase_index.py trend breakout_mid_50_100 phase1 breakout_1b_7030
  
  # Update Phase-2 index
  python scripts/update_phase_index.py trend breakout_mid_50_100 phase2 core_v3_tsb_phase2
  
  # Set sleeve status
  python scripts/update_phase_index.py trend persistence status "PARKED after Phase-1 fail"
        """
    )
    
    parser.add_argument("meta_sleeve", help="Meta-sleeve name (e.g., 'trend')")
    parser.add_argument("sleeve_name", help="Atomic sleeve name (e.g., 'breakout_mid_50_100')")
    parser.add_argument("phase", help="Phase name: 'phase0', 'phase1', 'phase2', or 'status'")
    parser.add_argument("value", nargs="?", help="For phase1/2: run_id. For status: status string")
    
    args = parser.parse_args()
    
    if args.phase == "status":
        if not args.value:
            parser.error("status requires a value (status string)")
        set_sleeve_status(args.meta_sleeve, args.sleeve_name, args.value)
        print(f"✓ Updated status for {args.meta_sleeve}/{args.sleeve_name}: {args.value}")
    elif args.phase in ["phase1", "phase2"]:
        if not args.value:
            parser.error(f"{args.phase} requires a value (run_id)")
        update_phase_index(args.meta_sleeve, args.sleeve_name, args.phase, run_id=args.value)
        print(f"✓ Updated {args.phase} index for {args.meta_sleeve}/{args.sleeve_name}: {args.value}")
    elif args.phase == "phase0":
        # Phase-0 is auto-updated by sanity check scripts
        print("Phase-0 index is automatically updated by sanity check scripts.")
        print("Use the sanity check script to update Phase-0.")
    else:
        parser.error(f"Unknown phase: {args.phase}. Use: phase0, phase1, phase2, or status")


if __name__ == "__main__":
    main()

