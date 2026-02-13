# Project Hub (Static HTML Site)

A minimal, local-first static HTML "Project Hub" for pinned runs, ops commands, and SOT documentation.

## How to Build

```bash
python3 scripts/site/build_site.py
```

Takes seconds. Output is written to `docs/site/`.

## How to Open

**macOS:**
```bash
open docs/site/index.html
```

**Windows:**
```bash
start docs/site/index.html
```

**Linux:**
```bash
xdg-open docs/site/index.html
```

Or open `docs/site/index.html` directly in your browser (file:// protocol).

## How to Refresh Pinned Runs

1. Edit `configs/pinned_runs.yaml` to add/update runs.
2. Add or update summary markdown in `docs/pinned/<run_id>.md`.
3. Optionally run `python scripts/site/extract_metrics.py --all` to refresh metrics from local run artifacts (if present).
4. Rerun `python3 scripts/site/build_site.py`.

## What This Is (and Isn't)

- **Static HTML.** It does NOT auto-refresh. Refresh by rerunning `build_site.py`.
- **Read-only.** It links to scripts and shows copy-paste commands; it does NOT execute them.
- **Local-first.** No server required. Open files directly (file://).
- **Not a control plane.** DB refreshes, heavy scripts, and ops execution are done via CLI/CI, not from the browser.
