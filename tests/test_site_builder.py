"""Lightweight tests for the static HTML Project Hub builder."""

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_pinned_runs_yaml_loads():
    import yaml
    path = PROJECT_ROOT / "configs" / "pinned_runs.yaml"
    assert path.exists(), "configs/pinned_runs.yaml must exist"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    runs = data.get("runs", [])
    assert len(runs) >= 1, "At least one pinned run required"
    for r in runs:
        assert "run_id" in r
        assert "label" in r
        assert "category" in r


def test_build_site_produces_index_html():
    import subprocess
    result = subprocess.run(
        ["python3", str(PROJECT_ROOT / "scripts" / "site" / "build_site.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"build_site.py failed: {result.stderr}"

    site_dir = PROJECT_ROOT / "docs" / "site"
    index_path = site_dir / "index.html"
    assert index_path.exists(), "index.html must be produced"


def test_run_pages_created():
    import yaml
    pinned_path = PROJECT_ROOT / "configs" / "pinned_runs.yaml"
    with open(pinned_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    runs = data.get("runs", [])

    site_dir = PROJECT_ROOT / "docs" / "site"
    runs_dir = site_dir / "runs"
    assert runs_dir.exists()

    for r in runs:
        rid = r["run_id"]
        run_page = runs_dir / f"{rid}.html"
        assert run_page.exists(), f"Run page {rid}.html must exist"


def test_vrp_drift_assertions():
    """Assert VRP atomic sleeves cannot appear at metasleeve level."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_site", PROJECT_ROOT / "scripts" / "site" / "build_site.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bad = [
        {"name": "vrp", "atomic_sleeves": [{"name": "vrp_core", "weight": 0.1}]},
        {"name": "vrp_core", "weight": 0.5},
    ]
    with pytest.raises(AssertionError, match="vrp_core.*must not appear"):
        mod._normalize_metasleeves(bad)


def test_docs_pages_created():
    SOT_DOCS = [
        "STRATEGY",
        "DIAGNOSTICS",
        "PROCEDURES",
        "SYSTEM_CONSTRUCTION",
        "ROADMAP",
    ]
    site_docs = PROJECT_ROOT / "docs" / "site" / "docs"
    assert site_docs.exists()

    for docname in SOT_DOCS:
        doc_path = PROJECT_ROOT / "docs" / "SOTs" / f"{docname}.md"
        if doc_path.exists():
            html_path = site_docs / f"{docname}.html"
            assert html_path.exists(), f"Doc page {docname}.html must exist"
