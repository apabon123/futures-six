# Phase Index

This directory contains canonical references to Phase-0, Phase-1, and Phase-2 results for each atomic sleeve.

## Structure

```
phase_index/
  {meta_sleeve}/
    {sleeve_name}/
      phase0.txt      # Points to latest Phase-0 results
      phase1.txt      # Points to Phase-1 run_id
      phase2.txt      # Points to Phase-2 run_id
      status.txt      # Optional: status for parked sleeves
```

## Usage

### Finding Canonical Results

**Phase-0**: Read `phase0.txt` to get the path to `latest/` directory:
```
sanity_checks/trend/breakout_mid_50_100/latest/
```

**Phase-1/2**: Read `phase1.txt` or `phase2.txt` to get the run_id, then look in:
```
reports/runs/{run_id}/
```

### Example

For `trend/breakout_mid_50_100`:
- `phase0.txt` → `sanity_checks/trend/breakout_mid_50_100/latest/`
- `phase1.txt` → `breakout_1b_7030`
- `phase2.txt` → `core_v3_tsb_phase2`

### Status Files

For parked sleeves (e.g., `persistence`):
- `status.txt` contains the status (e.g., "PARKED after Phase-1 fail")
- Phase files may point to last failed run for reference

## Updating

**Phase-0**: Automatically updated by sanity check scripts when they run.

**Phase-1/2**: Update manually after running Phase-1/2 backtests:
```bash
python scripts/update_phase_index.py trend breakout_mid_50_100 phase1 breakout_1b_7030
python scripts/update_phase_index.py trend breakout_mid_50_100 phase2 core_v3_tsb_phase2
```

**Status**: Set when a sleeve is parked:
```bash
python scripts/update_phase_index.py trend persistence status "PARKED after Phase-1 fail"
```

