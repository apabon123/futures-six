#!/usr/bin/env python3
"""
Build the static HTML Project Hub.

Generates docs/site/ with:
- index.html (Project Hub home)
- runs/index.html (Pinned Runs Dashboard)
- runs/<run_id>.html (Run detail pages)
- ops/index.html (Ops hub: gates, preflight, run commands)
- docs/index.html (Docs hub)
- docs/<docname>.html (Rendered SOT markdown)

Usage:
    python scripts/site/build_site.py

Output: docs/site/ (tracked in git)
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

# Markdown required (fail fast)
def _get_markdown():
    try:
        import markdown
        return markdown.markdown
    except ImportError:
        print("Missing dependency: markdown. Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

def _get_jinja():
    try:
        from jinja2 import Environment, BaseLoader
        return Environment(loader=BaseLoader())
    except ImportError:
        return None


# Embedded CSS (no external CDN, no JS)
SITE_CSS = """
:root { --bg: #1a1b26; --fg: #c0caf5; --muted: #565f89; --accent: #7aa2f7; }
body { font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--fg); margin: 0; padding: 1.5rem 2rem; line-height: 1.6; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
nav { display: flex; gap: 1.5rem; margin-bottom: 2rem; border-bottom: 1px solid var(--muted); padding-bottom: 1rem; }
nav a { font-weight: 500; }
h1, h2, h3 { color: var(--fg); margin-top: 1.5rem; }
h1 { font-size: 1.75rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid var(--muted); padding: 0.5rem 0.75rem; text-align: left; }
th { background: rgba(122,162,247,0.15); }
.badge { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.85rem; }
.badge-production { background: #1f7a1f; color: #fff; }
.badge-engine_quality { background: #1f5a7a; color: #fff; }
.badge-integration { background: #5a3a7a; color: #fff; }
.badge-diagnostic { background: #7a5a1a; color: #fff; }
.card { background: rgba(48,52,70,0.6); border: 1px solid var(--muted); border-radius: 8px; padding: 1rem; margin: 1rem 0; }
.card h3 { margin-top: 0; }
.card a { font-weight: 500; }
.muted { color: var(--muted); font-size: 0.9rem; }
pre, code { background: rgba(0,0,0,0.3); padding: 0.2rem 0.4rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }
pre { padding: 1rem; overflow-x: auto; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--muted); color: var(--muted); font-size: 0.85rem; }
"""


SOT_DOCS = [
    ("STRATEGY", "docs/SOTs/STRATEGY.md"),
    ("DIAGNOSTICS", "docs/SOTs/DIAGNOSTICS.md"),
    ("PROCEDURES", "docs/SOTs/PROCEDURES.md"),
    ("SYSTEM_CONSTRUCTION", "docs/SOTs/SYSTEM_CONSTRUCTION.md"),
    ("ROADMAP", "docs/SOTs/ROADMAP.md"),
]


# Attribution JSON uses engine names; map to display names for consistency with hierarchy
ATTRIBUTION_DISPLAY_ALIASES = {
    "vrp_combined": "vrp",
    "vrp_core_meta": "vrp_core",
    "vrp_convergence_meta": "vrp_convergence",
    "vrp_alt_meta": "vrp_alt",
}


def _normalize_metasleeves(metasleeves: list) -> list:
    """Normalize metasleeves: derive VRP weight from atomic weights (single source of truth)."""
    out = []
    for s in metasleeves:
        entry = dict(s)
        atomics = s.get("atomic_sleeves", [])
        if atomics and isinstance(atomics[0], dict):
            # Atomic-only format: weight = sum of atomic weights
            total = sum(a.get("weight", 0) for a in atomics)
            entry["weight"] = total
            entry["_atomic_names"] = [a.get("name", "") for a in atomics]
        elif atomics and isinstance(atomics[0], str):
            entry["_atomic_names"] = list(atomics)
        else:
            entry["_atomic_names"] = []
        out.append(entry)
    return out


def load_pinned_runs() -> list:
    path = PROJECT_ROOT / "configs" / "pinned_runs.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("runs", [])


def load_metrics(run_id: str) -> Optional[dict]:
    path = PROJECT_ROOT / "docs" / "pinned" / f"{run_id}.metrics.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_metric(metrics: Optional[dict], run_id: str, field: str) -> Optional[float]:
    """Get metric value, supporting max_drawdown as alias for maxdd. Print warning if key missing."""
    if not metrics:
        return None
    # maxdd can be stored as maxdd or max_drawdown
    key = field
    if field == "maxdd":
        key = "maxdd" if "maxdd" in metrics else "max_drawdown"
    v = metrics.get(key)
    if v is None and field in ("cagr", "sharpe", "vol", "maxdd"):
        print(f"[site] metrics key missing for {run_id}: {field}", file=sys.stderr)
    return v


def load_attribution(run_id: str) -> Optional[dict]:
    path = PROJECT_ROOT / "docs" / "pinned" / f"{run_id}.attribution.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_summary_md(rel_path: str) -> str:
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def md_to_html(md: str) -> str:
    fn = _get_markdown()
    return fn(md)


def rewrite_sot_links(html: str) -> str:
    """Rewrite SOT .md links to .html so they resolve in the static site.
    Handles: docs/SOTs/X.md, SOTs/X.md, ./docs/SOTs/X.md, ../docs/SOTs/X.md,
    and any of the above with #anchors -> X.html or X.html#section.
    """
    return re.sub(
        r'href="[^"]*SOTs/(\w+)\.md(#[^"]*)?"',
        r'href="\1.html\2"',
        html,
    )


def render_page(title: str, body: str, nav_links: list) -> str:
    nav = " | ".join(f'<a href="{href}">{label}</a>' for label, href in nav_links)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Futures-Six Project Hub</title>
<style>{SITE_CSS}</style>
</head>
<body>
<nav>{nav}</nav>
{body}
<footer>Last built: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</footer>
</body>
</html>"""


def build_index(site_dir: Path) -> None:
    nav = [
        ("Home", "index.html"),
        ("Runs", "runs/index.html"),
        ("Ops", "ops/index.html"),
        ("Docs", "docs/index.html"),
    ]
    body = """
<h1>Futures-Six Project Hub</h1>
<p>Static, local-first hub for pinned runs, ops commands, and SOT documentation.</p>
<ul>
  <li><a href="runs/index.html">Pinned Runs Dashboard</a> — Core v8/v9, Phase-3 baselines, VRP canonical, etc.</li>
  <li><a href="ops/index.html">Ops Hub</a> — Gates, preflight, run scripts.</li>
  <li><a href="docs/index.html">Docs Hub</a> — SOTs and key docs rendered to HTML.</li>
</ul>
<p class="muted">No server required. Open index.html directly (file://). Refresh by rerunning: python scripts/site/build_site.py</p>
"""
    html = render_page("Project Hub", body, nav)
    (site_dir / "index.html").write_text(html, encoding="utf-8")


def build_runs_dashboard(site_dir: Path, runs: list) -> None:
    nav = [
        ("Home", "../index.html"),
        ("Runs", "index.html"),
        ("Ops", "../ops/index.html"),
        ("Docs", "../docs/index.html"),
    ]
    rows = []
    for r in runs:
        rid = r["run_id"]
        metrics = load_metrics(rid)
        cagr = _get_metric(metrics, rid, "cagr")
        sharpe = _get_metric(metrics, rid, "sharpe")
        vol = _get_metric(metrics, rid, "vol")
        maxdd = _get_metric(metrics, rid, "maxdd")
        cagr_s = f"{float(cagr)*100:.2f}%" if cagr is not None else "N/A"
        sharpe_s = f"{float(sharpe):.3f}" if sharpe is not None else "N/A"
        vol_s = f"{float(vol)*100:.2f}%" if vol is not None else "N/A"
        maxdd_s = f"{float(maxdd)*100:.2f}%" if maxdd is not None else "N/A"

        display = r.get("display_name") or r.get("label") or rid
        cat = r.get("category", "integration")
        window = r.get("window", {})
        wstr = f"{window.get('start','')} → {window.get('end','')}"

        rows.append(f"""
<tr>
  <td><a href="{rid}.html">{display}</a></td>
  <td><span class="badge badge-{cat}">{cat}</span></td>
  <td>{wstr}</td>
  <td>{cagr_s}</td>
  <td>{sharpe_s}</td>
  <td>{vol_s}</td>
  <td>{maxdd_s}</td>
  <td>{r.get('config_path','')}</td>
</tr>""")

    table = "<table><thead><tr><th>Run</th><th>Category</th><th>Window</th><th>CAGR</th><th>Sharpe</th><th>Vol</th><th>MaxDD</th><th>Config</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

    cards = []
    for r in runs:
        rid = r["run_id"]
        display = r.get("display_name") or r.get("label") or rid
        metasleeves = _normalize_metasleeves(r.get("sleeves", {}).get("metasleeves", []))
        # Dashboard shows metasleeves only (VRP once, weight derived from atomics)
        sleeves_s = ", ".join(f"{s.get('name','')} ({s.get('weight',0)})" for s in metasleeves[:5])
        if len(metasleeves) > 5:
            sleeves_s += "…"
        cards.append(f"""
<div class="card">
  <h3><a href="{rid}.html">{display}</a></h3>
  <p class="muted">{r.get('description','')}</p>
  <p><strong>Sleeves:</strong> {sleeves_s}</p>
</div>""")

    body = """
<h1>Pinned Runs Dashboard</h1>
<p>Core v8/v9, Phase-3 baselines, VRP canonical, trend-only/trend+VRP.</p>
<h2>Table View</h2>
""" + table + """
<h2>Card View</h2>
""" + "\n".join(cards)

    html = render_page("Pinned Runs", body, nav)
    (site_dir / "runs").mkdir(parents=True, exist_ok=True)
    (site_dir / "runs" / "index.html").write_text(html, encoding="utf-8")


def _render_attribution(attribution: Optional[dict]) -> str:
    """Render attribution section HTML from docs/pinned/<run_id>.attribution.json."""
    if not attribution:
        return "<p>Attribution not available for this run.</p>"

    parts = []
    # Consistency check
    cc = attribution.get("consistency_check", {})
    passed = cc.get("passed", False)
    max_residual = cc.get("max_abs_daily_residual", cc.get("max_abs_daily_residual_active"))
    consistency = f"<p><strong>Consistency check:</strong> {'Pass' if passed else 'Fail'}"
    if max_residual is not None:
        consistency += f" (max residual: {max_residual:.2e})"
    consistency += "</p>"
    parts.append(consistency)

    # Metasleeve summary table (alias engine names -> display names for consistency)
    meta = attribution.get("metasleeve_summary", [])
    if meta:
        rows = "".join(
            f"<tr><td>{ATTRIBUTION_DISPLAY_ALIASES.get(r.get('metasleeve',''), r.get('metasleeve',''))}</td><td>{r.get('cum_return','')}</td></tr>"
            for r in meta
        )
        parts.append("<h4>Metasleeve contributions</h4><table><thead><tr><th>Metasleeve</th><th>Cum return</th></tr></thead><tbody>" + rows + "</tbody></table>")

    # Atomic sleeve summary table (alias vrp_*_meta -> vrp_* for display)
    atomic = attribution.get("atomic_summary", attribution.get("per_sleeve", {}))
    if isinstance(atomic, dict) and atomic:
        rows = "".join(
            f"<tr><td>{ATTRIBUTION_DISPLAY_ALIASES.get(s, s)}</td><td>{d.get('cum_return','') if isinstance(d, dict) else d}</td></tr>"
            for s, d in atomic.items()
        )
        parts.append("<h4>Atomic sleeve contributions</h4><table><thead><tr><th>Sleeve</th><th>Cum return</th></tr></thead><tbody>" + rows + "</tbody></table>")

    # Correlation matrix path
    corr_path = attribution.get("correlation_matrix_path", "")
    if corr_path:
        parts.append(f"<p class='muted'>Correlation matrix: {corr_path}</p>")

    parts.append("<p class='muted'>Generated from portfolio-consistent attribution system.</p>")
    return "\n".join(parts)


def build_run_detail(site_dir: Path, run: dict) -> None:
    rid = run["run_id"]
    display = run.get("display_name") or run.get("label") or rid
    nav = [
        ("Home", "../index.html"),
        ("Runs", "index.html"),
        ("Ops", "../ops/index.html"),
        ("Docs", "../docs/index.html"),
    ]
    summary_path = run.get("artifacts", {}).get("summary_md", "")
    summary_md = load_summary_md(summary_path) if summary_path else ""
    summary_html = md_to_html(summary_md) if summary_md else "<p>No summary available.</p>"

    # Headline metrics (cagr, sharpe, vol, maxdd) with warning for missing keys
    metrics = load_metrics(rid)
    metrics_block = ""
    if metrics:
        m = metrics
        metrics_block = "<h3>Headline Metrics</h3><ul>"
        for k in ["cagr", "vol", "sharpe", "maxdd", "turnover", "hit_rate"]:
            v = _get_metric(m, rid, k) if k in ("cagr", "sharpe", "vol", "maxdd") else m.get(k)
            if v is not None:
                if isinstance(v, float) and k in ("cagr", "vol", "maxdd") and abs(v) < 1:
                    v = f"{v*100:.2f}%"
                elif isinstance(v, float) and k == "sharpe":
                    v = f"{v:.3f}"
                metrics_block += f"<li><strong>{k}</strong>: {v}</li>"
        metrics_block += "</ul>"

    metasleeves = _normalize_metasleeves(run.get("sleeves", {}).get("metasleeves", []))
    sleeves_rows = []
    for s in metasleeves:
        name = s.get("name", "")
        weight = s.get("weight", "")
        atomics = s.get("_atomic_names", [])
        atomics_str = ", ".join(str(a) for a in atomics) if atomics else "—"
        sleeves_rows.append(f"<tr><td>{name}</td><td>{weight}</td><td>{atomics_str}</td></tr>")
    sleeves_table = (
        "<table><thead><tr><th>Metasleeve</th><th>Weight</th><th>Atomic sleeves</th></tr></thead><tbody>"
        + "".join(sleeves_rows)
        + "</tbody></table>"
    ) if sleeves_rows else ""

    reproduce = summary_md
    m = re.search(r"```bash\n(.*?)\n```", reproduce, re.DOTALL)
    reproduce_block = f"<pre><code>{m.group(1).strip()}</code></pre>" if m else "<p>See summary for reproduce command.</p>"

    # Attribution section
    attribution = load_attribution(rid)
    attribution_block = _render_attribution(attribution)

    body = f"""
<h1>{display}</h1>
<p class="muted">Run ID: {rid} | Config: {run.get('config_path','')} | Window: {run.get('window',{}).get('start','')} → {run.get('window',{}).get('end','')}</p>

<h3>Run description</h3>
{summary_html}

{metrics_block}
<h3>Sleeves</h3>
{sleeves_table}

<h3>Attribution</h3>
{attribution_block}

<h3>How to Reproduce</h3>
{reproduce_block}
"""
    html = render_page(display, body, nav)
    (site_dir / "runs" / f"{rid}.html").write_text(html, encoding="utf-8")


def build_ops_hub(site_dir: Path) -> None:
    nav = [
        ("Home", "../index.html"),
        ("Runs", "../runs/index.html"),
        ("Ops", "index.html"),
        ("Docs", "../docs/index.html"),
    ]
    body = """
<h1>Ops Hub</h1>
<p>Operational scripts: gates, preflight, run canonical scripts. Copy-paste commands below.</p>

<h2>VRP Refresh Gate</h2>
<pre><code>python scripts/gates/vrp_refresh_gate.py
python scripts/gates/vrp_refresh_gate.py --profile phase4_vrp_baseline_v1
python scripts/gates/vrp_refresh_gate.py --config_path configs/phase4_vrp_baseline_v1.yaml</code></pre>

<h2>Preflight Coverage Checks</h2>
<pre><code>python scripts/preflight/check_window_coverage.py --start 2024-03-01 --end 2024-04-30
python scripts/preflight/check_window_coverage.py --start 2020-01-01 --end 2024-10-31</code></pre>

<h2>Run Canonical Scripts</h2>
<pre><code>python scripts/runs/run_vrp_canonical_2020_2024.py
python run_strategy.py --start 2020-01-01 --end 2024-10-31 --run_id trend_only_canonical_2020_2024 --config_path configs/phase4_trend_only_canonical_v1.yaml --strict_universe
python run_strategy.py --start 2020-01-01 --end 2024-10-31 --run_id trend_plus_vrp_canonical_2020_2024 --config_path configs/phase4_trend_plus_vrp_canonical_v1.yaml --strict_universe</code></pre>

<p class="muted">This hub is read-only. Execute commands from your terminal.</p>
"""
    html = render_page("Ops Hub", body, nav)
    (site_dir / "ops").mkdir(parents=True, exist_ok=True)
    (site_dir / "ops" / "index.html").write_text(html, encoding="utf-8")


def build_docs_hub(site_dir: Path) -> None:
    nav = [
        ("Home", "../index.html"),
        ("Runs", "../runs/index.html"),
        ("Ops", "../ops/index.html"),
        ("Docs", "index.html"),
    ]
    links = []
    for docname, rel_path in SOT_DOCS:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            links.append(f'<li><a href="{docname}.html">{docname}</a></li>')
    body = """
<h1>Docs Hub</h1>
<p>SOTs and key docs rendered to HTML.</p>
<ul>
""" + "\n".join(links) + """
</ul>
"""
    html = render_page("Docs Hub", body, nav)
    (site_dir / "docs").mkdir(parents=True, exist_ok=True)
    (site_dir / "docs" / "index.html").write_text(html, encoding="utf-8")


def build_doc_pages(site_dir: Path) -> None:
    nav_base = [
        ("Home", "../index.html"),
        ("Runs", "../runs/index.html"),
        ("Ops", "../ops/index.html"),
        ("Docs", "index.html"),
    ]
    for docname, rel_path in SOT_DOCS:
        p = PROJECT_ROOT / rel_path
        if not p.exists():
            continue
        md = p.read_text(encoding="utf-8")
        body_html = rewrite_sot_links(md_to_html(md))
        body = f"<h1>{docname}</h1>{body_html}"
        html = render_page(docname, body, nav_base)
        (site_dir / "docs" / f"{docname}.html").write_text(html, encoding="utf-8")


def main():
    site_dir = PROJECT_ROOT / "docs" / "site"
    site_dir.mkdir(parents=True, exist_ok=True)

    runs = load_pinned_runs()
    if not runs:
        print("No pinned runs in configs/pinned_runs.yaml")
        sys.exit(1)

    build_index(site_dir)
    build_runs_dashboard(site_dir, runs)
    for r in runs:
        build_run_detail(site_dir, r)
    build_ops_hub(site_dir)
    build_docs_hub(site_dir)
    build_doc_pages(site_dir)

    print(f"Built site at {site_dir}")
    print(f"  - index.html")
    print(f"  - runs/index.html, runs/<run_id>.html ({len(runs)} runs)")
    print(f"  - ops/index.html")
    print(f"  - docs/index.html, docs/<doc>.html ({len(SOT_DOCS)} docs)")


if __name__ == "__main__":
    main()
