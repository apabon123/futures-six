"""
Attribution governance: consistency status thresholds (PASS / WARN / FAIL).

Used for run-page display and ATTRIBUTION_SUMMARY; not for unit-test strictness.
"""

# Governance thresholds (max absolute daily residual)
ATTR_TOL_PASS = 1e-5   # status PASS if max_abs_residual <= this
ATTR_TOL_WARN = 1e-4   # status WARN if residual in (ATTR_TOL_PASS, ATTR_TOL_WARN]; FAIL if > ATTR_TOL_WARN


def attribution_status_from_residual(max_abs_residual: float) -> str:
    """Return PASS, WARN, or FAIL from max absolute daily residual."""
    if max_abs_residual <= ATTR_TOL_PASS:
        return "PASS"
    if max_abs_residual <= ATTR_TOL_WARN:
        return "WARN"
    return "FAIL"
