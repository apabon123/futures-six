# Reports Directory Structure

## Overview

The reports directory uses a canonical structure that makes it easy to find the current Phase-0, Phase-1, and Phase-2 results for each atomic sleeve without cluttering the view with old timestamp folders.

## Directory Layout

```
reports/
  sanity_checks/              # Phase-0 raw logs (all runs)
    {meta_sleeve}/
      {sleeve_name}/
        archive/
          {timestamp}/        # All historical Phase-0 runs
        latest/               # Canonical Phase-0 results (always current)
        latest_failed/        # Optional: last failing Phase-0 (for parked sleeves)
  
  phase_index/                # Canonical references to current runs
    {meta_sleeve}/
      {sleeve_name}/
        phase0.txt           # Points to latest/ directory
        phase1.txt           # Points to Phase-1 run_id
        phase2.txt           # Points to Phase-2 run_id
        status.txt           # Optional: status for parked sleeves
  
  runs/                       # Phase-1/2 backtest runs (unchanged)
    {run_id}/
      ...
```

## Example: Trend Breakout (50-100d)

### Phase-0 Structure

```
reports/
  sanity_checks/
    trend/
      breakout_mid_50_100/
        archive/
          20251118_225521/    # Historical run
          20251119_091500/    # Another historical run
        latest/               # ← Current canonical Phase-0
          portfolio_returns.csv
          equity_curve.csv
          equity_curve.png
          per_asset_stats.csv
          meta.json
          return_histogram.png
```

### Phase Index

```
reports/
  phase_index/
    trend/
      breakout_mid_50_100/
        phase0.txt           # Contains: "sanity_checks/trend/breakout_mid_50_100/latest"
        phase1.txt           # Contains: "breakout_1b_7030"
        phase2.txt           # Contains: "core_v3_tsb_phase2"
```

## Finding Canonical Results

### Quick Answer: "What's the current Phase-0/1/2 for this sleeve?"

1. **Open** `reports/phase_index/{meta_sleeve}/{sleeve_name}/`
2. **Read** `phase0.txt`, `phase1.txt`, or `phase2.txt`
3. **Navigate** to the referenced location

### Phase-0

```python
from src.utils.phase_index import get_phase_path

# Get canonical Phase-0 path
phase0_path = get_phase_path("trend", "breakout_mid_50_100", "phase0")
# Returns: Path("reports/sanity_checks/trend/breakout_mid_50_100/latest")
```

### Phase-1/2

```python
from src.utils.phase_index import get_phase_path

# Get canonical Phase-1 path
phase1_path = get_phase_path("trend", "breakout_mid_50_100", "phase1")
# Returns: Path("reports/runs/breakout_1b_7030")
```

## How It Works

### Phase-0 (Sanity Checks)

When you run a sanity check script (e.g., `run_trend_breakout_mid_sanity.py`):

1. **Writes to archive**: All results go to `archive/{timestamp}/`
2. **Copies to latest**: Key files are copied to `latest/`
3. **Updates index**: `phase_index/.../phase0.txt` is updated automatically

**Files in `latest/`**:
- `portfolio_returns.csv`
- `equity_curve.csv`
- `equity_curve.png`
- `per_asset_stats.csv`
- `meta.json`
- `return_histogram.png`

### Phase-1/2 (Backtests)

After running a Phase-1/2 backtest:

1. **Run exists in** `reports/runs/{run_id}/`
2. **Update index manually**:
   ```bash
   python scripts/update_phase_index.py trend breakout_mid_50_100 phase1 breakout_1b_7030
   python scripts/update_phase_index.py trend breakout_mid_50_100 phase2 core_v3_tsb_phase2
   ```

### Parked Sleeves

For sleeves that failed (e.g., `persistence`):

```
reports/
  sanity_checks/
    trend/
      persistence/
        archive/
          {timestamps}/
        latest_failed/        # Last failing run
  phase_index/
    trend/
      persistence/
        status.txt           # "PARKED after Phase-1 fail"
        phase0.txt           # Optional: points to latest_failed/
```

## Migration

To migrate existing runs to the new structure:

```bash
python scripts/migrate_phase_structure.py
```

This script:
1. Moves existing timestamped runs to `archive/`
2. Copies the most recent passing run to `latest/`
3. Creates `phase_index` entries

## Benefits

✅ **Clean View**: No timestamp clutter at the top level  
✅ **Canonical Results**: Always know which run is "the one"  
✅ **Easy Navigation**: Quick answer to "what's the current Phase-X?"  
✅ **Audit Trail**: All historical runs preserved in `archive/`  
✅ **Flexible**: Can delete old archives without breaking references  

## Maintenance

### Deleting Old Archives

You can safely delete old `archive/{timestamp}/` directories. The `latest/` directory and `phase_index` entries are independent.

### Updating Phase-1/2 Index

After running a new Phase-1/2 backtest, update the index:

```bash
python scripts/update_phase_index.py {meta_sleeve} {sleeve_name} {phase} {run_id}
```

### Setting Sleeve Status

For parked sleeves:

```bash
python scripts/update_phase_index.py trend persistence status "PARKED after Phase-1 fail"
```

## Related Documentation

- **`TREND_RESEARCH.md`**: Research notebook with Phase-0/1/2 results
- **`DIAGNOSTICS.md`**: Diagnostics framework documentation
- **`PROCEDURES.md`**: Sleeve lifecycle procedures

