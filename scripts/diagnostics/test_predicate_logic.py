"""Test predicate logic with actual values from debug log."""
dd_level = -0.180210
dd_stress_enter = -0.18
dd_crisis_enter = -0.20

# Test comparisons
stress_pred = dd_level <= dd_stress_enter  # -0.180210 <= -0.18
crisis_pred = dd_level <= dd_crisis_enter  # -0.180210 <= -0.20

print(f"dd_level: {dd_level}")
print(f"dd_stress_enter: {dd_stress_enter}")
print(f"dd_crisis_enter: {dd_crisis_enter}")
print(f"\nstress_pred (-0.180210 <= -0.18): {stress_pred}")
print(f"crisis_pred (-0.180210 <= -0.20): {crisis_pred}")

# For negative numbers: more negative = worse
# -0.180210 is MORE negative than -0.18, so -0.180210 < -0.18 (TRUE)
# -0.180210 is LESS negative than -0.20, so -0.180210 > -0.20 (FALSE)

print(f"\nMathematically:")
print(f"  -0.180210 < -0.18: {(-0.180210 < -0.18)}")
print(f"  -0.180210 <= -0.18: {(-0.180210 <= -0.18)}")
print(f"  -0.180210 < -0.20: {(-0.180210 < -0.20)}")
print(f"  -0.180210 <= -0.20: {(-0.180210 <= -0.20)}")
