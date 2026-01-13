"""
Run Canonical Dashboard

Launches the Streamlit dashboard for interactive visualization.

Usage:
    python scripts/dashboards/run_canonical_dashboard.py [--port PORT] [--run_id RUN_ID]

The dashboard reads artifacts from reports/runs/{run_id}/ and provides:
1. Equity + Drawdown chart
2. Exposure by sleeve over time
3. Position-level view (per asset for any date)
4. Allocator state timeline

Key Principle: Dashboard reads artifacts only. No strategy logic computation.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch Canonical Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The dashboard provides interactive visualization of backtest results.

Views:
1. Equity + Drawdown
2. Exposure Over Time (raw/post-allocator)
3. Position-Level View (per asset for any date)
4. Allocator State Timeline (regime/scalars/drawdown)

Examples:
  # Launch dashboard (default port 8501)
  python scripts/dashboards/run_canonical_dashboard.py
  
  # Launch on custom port
  python scripts/dashboards/run_canonical_dashboard.py --port 8502
  
Note: Requires streamlit and plotly to be installed.
        """
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run dashboard on (default: 8501)'
    )
    
    parser.add_argument(
        '--run_id',
        type=str,
        default=None,
        help='Optional: Default run ID to load (default: most recent)'
    )
    
    args = parser.parse_args()
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Error: streamlit is not installed.")
        print("Install it with: pip install streamlit")
        return 1
    
    # Check if plotly is installed
    try:
        import plotly
    except ImportError:
        print("Error: plotly is not installed.")
        print("Install it with: pip install plotly")
        return 1
    
    # Get path to dashboard script
    dashboard_script = project_root / "src" / "dashboards" / "canonical_dashboard.py"
    
    if not dashboard_script.exists():
        print(f"Error: Dashboard script not found: {dashboard_script}")
        return 1
    
    # Launch streamlit
    import subprocess
    
    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        str(dashboard_script),
        "--server.port", str(args.port)
    ]
    
    print(f"Launching Canonical Dashboard on port {args.port}...")
    print(f"Dashboard will be available at: http://localhost:{args.port}")
    print("Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error launching dashboard: {e}")
        return 1
    except FileNotFoundError:
        print("Error: streamlit command not found.")
        print("Make sure streamlit is installed: pip install streamlit")
        return 1


if __name__ == "__main__":
    sys.exit(main())
