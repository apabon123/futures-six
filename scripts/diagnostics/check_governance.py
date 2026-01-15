"""Quick script to check governance fields in canonical diagnostics."""
import sys
import json
from pathlib import Path

run_id = sys.argv[1]
run_dir = Path(f"reports/runs/{run_id}")
diag_file = run_dir / "canonical_diagnostics.json"

if not diag_file.exists():
    print(f"ERROR: {diag_file} not found")
    sys.exit(1)

with open(diag_file, 'r', encoding='utf-8') as f:
    diag = json.load(f)

binding = diag.get('constraint_binding', {})

print("=" * 80)
print("GOVERNANCE STATUS CHECK")
print("=" * 80)
print(f"\nRun ID: {run_id}")
print()

# Policy
print("Policy (Phase 3A):")
print(f"  enabled: {binding.get('policy_enabled', False)}")
print(f"  effective: {binding.get('policy_effective', False)}")
print(f"  inert: {binding.get('policy_inert', False)}")
if binding.get('policy_inert'):
    print(f"  inert_reason: {binding.get('policy_inert_reason', 'N/A')}")
print()

# Risk Targeting
print("Risk Targeting (Layer 5):")
print(f"  enabled: {binding.get('rt_enabled', False)}")
print(f"  effective: {binding.get('rt_effective', False)}")
print(f"  has_teeth: {binding.get('rt_has_teeth', False)}")
print(f"  inert: {binding.get('rt_inert', False)}")
if binding.get('rt_inert'):
    print(f"  inert_reason: {binding.get('rt_inert_reason', 'N/A')}")
if 'rt_multiplier_p50' in binding:
    print(f"  multiplier_p50: {binding.get('rt_multiplier_p50', 'N/A'):.3f}")
    print(f"  multiplier_at_cap_pct: {binding.get('rt_multiplier_at_cap_pct', 0.0):.1f}%")
print()

# Allocator v1
print("Allocator v1 (Layer 6):")
print(f"  enabled: {binding.get('alloc_v1_enabled', False)}")
print(f"  effective: {binding.get('alloc_v1_effective', False)}")
print(f"  has_teeth: {binding.get('alloc_v1_has_teeth', False)}")
print(f"  inert: {binding.get('alloc_v1_inert', False)}")
if binding.get('alloc_v1_inert'):
    print(f"  inert_reason: {binding.get('alloc_v1_inert_reason', 'N/A')}")
if 'alloc_v1_regime_normal_pct' in binding:
    print(f"  regime_normal_pct: {binding.get('alloc_v1_regime_normal_pct', 0.0):.1f}%")
    print(f"  regime_stress_pct: {binding.get('alloc_v1_regime_stress_pct', 0.0):.1f}%")
if 'alloc_v1_scalar_p50' in binding:
    print(f"  scalar_p50: {binding.get('alloc_v1_scalar_p50', 'N/A'):.3f}")
    print(f"  scalar_at_min_pct: {binding.get('alloc_v1_scalar_at_min_pct', 0.0):.1f}%")
print()

# Summary
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

all_valid = True
if binding.get('policy_enabled') and not binding.get('policy_effective'):
    print("[FAIL] Policy enabled but not effective")
    all_valid = False
elif binding.get('policy_enabled'):
    print("[PASS] Policy effective")

if binding.get('rt_enabled') and not binding.get('rt_effective'):
    print("[FAIL] Risk Targeting enabled but not effective")
    all_valid = False
elif binding.get('rt_enabled'):
    print("[PASS] Risk Targeting effective")

if binding.get('alloc_v1_enabled') and not binding.get('alloc_v1_effective'):
    print("[FAIL] Allocator v1 enabled but not effective")
    all_valid = False
elif binding.get('alloc_v1_enabled'):
    print("[PASS] Allocator v1 effective")

if all_valid:
    print("\n[SUCCESS] All enabled layers are effective")
else:
    print("\n[FAILURE] Some layers are inert")
    sys.exit(1)
