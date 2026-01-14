import json
from pathlib import Path
import pandas as pd
import numpy as np

RUN_ID = "canonical_frozen_stack_precomputed_20260113_123354"
BASE_DIR = Path("reports/runs")  # adjust if your repo uses a different root
RUN_DIR = BASE_DIR / RUN_ID

def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _print_header(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

def main():
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Run dir not found: {RUN_DIR.resolve()}")

    _print_header(f"ENGINE POLICY AUDIT — {RUN_ID}")
    print(f"Run dir: {RUN_DIR.resolve()}")

    # ---------------------------------------------------------------------
    # 1) META sanity check
    # ---------------------------------------------------------------------
    meta_path = RUN_DIR / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = _read_json(meta_path)
        _print_header("META.JSON (policy-related fields)")
        # Print a few fields if present, but don't crash if absent.
        keys_of_interest = [
            "run_id",
            "strategy_profile",
            "strategy_config_name",
            "config_hash",
            "engine_policy_source_run_id",
            "allocator_source_run_id",
            "start_date",
            "end_date",
            "effective_start_date",
        ]
        for k in keys_of_interest:
            if k in meta:
                print(f"{k}: {meta[k]}")
    else:
        print("WARNING: meta.json not found (this is unexpected for a canonical run).")

    # ---------------------------------------------------------------------
    # 2) Find policy/gate artifacts
    # ---------------------------------------------------------------------
    _print_header("SEARCHING FOR POLICY/GATE ARTIFACTS")
    patterns = [
        "*policy*.*",
        "*gate*.*",
        "*engine_policy*.*",
        "*multipliers*.*",
    ]
    found = []
    for pat in patterns:
        found.extend(sorted(RUN_DIR.glob(pat)))
    # De-dupe while preserving order
    seen = set()
    found_unique = []
    for p in found:
        if p not in seen:
            found_unique.append(p)
            seen.add(p)

    # Check specifically for the expected artifact
    expected_artifact = RUN_DIR / "engine_policy_applied_v1.csv"
    print(f"Expected artifact: engine_policy_applied_v1.csv")
    if expected_artifact.exists():
        print(f"[OK] FOUND: {expected_artifact.name}")
    else:
        print(f"[MISSING] {expected_artifact.name}")

    if not found_unique:
        print("\nNo files matched policy/gate patterns.")
        print("This does NOT prove policy was inactive — it may mean policy series is not being exported.")
    else:
        print(f"\nFound {len(found_unique)} file(s) matching patterns:")
        for p in found_unique:
            print(f"  - {p.name}")

    # Check source run if this is a precomputed run
    if meta_path.exists():
        source_run_id = meta.get("engine_policy_source_run_id")
        if source_run_id:
            source_run_dir = BASE_DIR / source_run_id
            _print_header(f"CHECKING SOURCE RUN: {source_run_id}")
            if source_run_dir.exists():
                source_artifact = source_run_dir / "engine_policy_applied_v1.csv"
                if source_artifact.exists():
                    print(f"[OK] FOUND in source run: {source_artifact.name}")
                else:
                    print(f"[MISSING] in source run: {source_artifact.name}")
                    print(f"   Source run dir: {source_run_dir.resolve()}")
                    print("   This is likely the root cause: source run doesn't have policy artifacts!")
            else:
                print(f"[ERROR] Source run directory does not exist: {source_run_dir}")

    # ---------------------------------------------------------------------
    # 3) If canonical_diagnostics.json exists, read any policy binding fields
    # ---------------------------------------------------------------------
    diag_json_path = RUN_DIR / "canonical_diagnostics.json"
    if diag_json_path.exists():
        diag = _read_json(diag_json_path)
        _print_header("CANONICAL_DIAGNOSTICS.JSON (policy binding hints if present)")
        # Print a few likely locations without assuming schema too hard.
        # You can expand these keys once we know the exact JSON structure.
        for k in ["constraint_binding", "constraints", "policy", "engine_policy", "binding_report"]:
            if k in diag:
                print(f"Found key: {k}")
                if k == "constraint_binding" and isinstance(diag[k], dict):
                    for subk, subv in diag[k].items():
                        print(f"  {subk}: {subv}")
    else:
        print("NOTE: canonical_diagnostics.json not found in run dir (but your README says it should exist).")

    # ---------------------------------------------------------------------
    # 4) Try to load a policy multipliers CSV if present and compute binding
    # ---------------------------------------------------------------------
    _print_header("POLICY BINDING COMPUTATION (from discovered CSVs)")

    # First, try the expected artifact name
    expected_csv = RUN_DIR / "engine_policy_applied_v1.csv"
    candidate_csvs = [expected_csv] if expected_csv.exists() else []
    
    # Also check discovered files
    candidate_csvs.extend([p for p in found_unique if p.suffix.lower() == ".csv" and p not in candidate_csvs])
    
    loaded_any = False

    for csv_path in candidate_csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"\nSkipping {csv_path.name}: could not read CSV ({e})")
            continue

        # Heuristics: find numeric columns that look like multipliers/scalars
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            continue

        # Some files might have a date column; try to parse it if present.
        date_col = None
        for c in df.columns:
            if c.lower() in ["date", "rebalance_date", "timestamp"]:
                date_col = c
                break

        if date_col is not None:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # For engine_policy_applied_v1.csv, the multiplier column is 'policy_multiplier_used'
        # Check for expected format first
        multiplier_col = None
        if 'policy_multiplier_used' in df.columns:
            multiplier_col = 'policy_multiplier_used'
        elif 'multiplier' in df.columns:
            multiplier_col = 'multiplier'
        elif len(numeric_cols) == 1:
            multiplier_col = numeric_cols[0]
        elif len(numeric_cols) > 1:
            # Use the first numeric column that looks like a multiplier
            multiplier_col = numeric_cols[0]

        if multiplier_col is None:
            print(f"\nSkipping {csv_path.name}: no multiplier column found")
            continue

        # Define "binding" as multiplier < 0.999 (tolerate float noise)
        # For binary gates {0,1}, this works too.
        tol = 1e-3
        bind_mask = df[multiplier_col] < (1.0 - tol)

        n = len(df)
        n_bind = int(bind_mask.sum())
        pct_bind = (n_bind / n * 100.0) if n else 0.0
        min_mult = float(df.loc[bind_mask, multiplier_col].min()) if n_bind else 1.0
        max_mult = float(df.loc[~bind_mask, multiplier_col].max()) if not bind_mask.all() else 1.0

        print(f"\nFile: {csv_path.name}")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        print(f"Multiplier column: {multiplier_col}")
        print(f"Total rows: {n}")
        print(f"Binding rows (multiplier < 1.0): {n_bind} ({pct_bind:.2f}%)")
        print(f"Min multiplier (when binding): {min_mult}")
        print(f"Max multiplier (when not binding): {max_mult}")

        if n_bind and date_col is not None:
            bind_dates = df.loc[bind_mask, date_col].dropna()
            if not bind_dates.empty:
                print(f"First bind date: {bind_dates.min().date()}")
                print(f"Last bind date : {bind_dates.max().date()}")

        # Engine-specific analysis if 'engine' column exists
        if 'engine' in df.columns:
            print("\nBinding by engine:")
            for engine in df['engine'].unique():
                engine_mask = df['engine'] == engine
                engine_bind = (engine_mask & bind_mask).sum()
                engine_total = engine_mask.sum()
                engine_pct = (engine_bind / engine_total * 100.0) if engine_total > 0 else 0.0
                engine_min_mult = float(df.loc[engine_mask, multiplier_col].min())
                print(f"  {engine}: {engine_bind}/{engine_total} ({engine_pct:.2f}%), min_mult={engine_min_mult:.4f}")

        # Show the 10 most severe rows (smallest multiplier) if binding exists
        if n_bind:
            worst = df.loc[bind_mask].sort_values(multiplier_col).head(10)
            cols_to_show = ([date_col] if date_col else []) + (['engine'] if 'engine' in df.columns else []) + [multiplier_col]
            print("\nWorst 10 binding rows (lowest multipliers):")
            print(worst[cols_to_show].to_string(index=False))

        loaded_any = True

    if not loaded_any:
        print("No readable policy/gate CSVs found to compute binding from.")
        print("If this is the case, the next step is to EXPORT policy multipliers at rebalances from ExecSim.")

    # ---------------------------------------------------------------------
    # 5) Optional cross-check: weights_raw vs weights_scaled presence
    # ---------------------------------------------------------------------
    _print_header("WEIGHTS ARTIFACT CROSS-CHECK")
    weights_raw = RUN_DIR / "weights_raw.csv"
    weights_scaled = RUN_DIR / "weights_scaled.csv"
    weights = RUN_DIR / "weights.csv"

    for p in [weights, weights_raw, weights_scaled]:
        if p.exists():
            print(f"Found: {p.name}")
        else:
            print(f"Missing: {p.name}")

    print("\nDone.")

if __name__ == "__main__":
    main()
