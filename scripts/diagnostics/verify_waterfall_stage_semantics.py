"""
Verify Phase 3B Waterfall Stage Semantics

Quick verification that:
- artifacts_only: post-RT == post-allocator (identity holds)
- traded: post-allocator differs from post-RT when scalars < 1.0
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def verify_stage_semantics(run_id: str, is_artifacts_only: bool):
    """Verify stage semantics for a baseline run."""
    run_dir = Path(f"reports/runs/{run_id}")
    waterfall_path = run_dir / 'waterfall_attribution.json'
    
    if not waterfall_path.exists():
        print(f"  [ERROR] Waterfall attribution not found: {waterfall_path}")
        return False
    
    with open(waterfall_path, 'r') as f:
        wf = json.load(f)
    
    portfolio_wf = wf.get('portfolio_waterfall', {})
    
    if 'post_rt' not in portfolio_wf or 'post_allocator' not in portfolio_wf:
        print(f"  [ERROR] Missing required stages in waterfall attribution")
        return False
    
    post_rt_metrics = portfolio_wf['post_rt'].get('metrics', {})
    post_alloc_metrics = portfolio_wf['post_allocator'].get('metrics', {})
    
    if is_artifacts_only:
        # Should be identical (identity check) - compare post_rt to post_allocator_base
        post_alloc_base_metrics = portfolio_wf.get('post_allocator_base', {}).get('metrics', {})
        
        rt_sharpe = post_rt_metrics.get('sharpe')
        alloc_base_sharpe = post_alloc_base_metrics.get('sharpe')
        rt_cagr = post_rt_metrics.get('cagr')
        alloc_base_cagr = post_alloc_base_metrics.get('cagr')
        
        sharpe_match = abs(rt_sharpe - alloc_base_sharpe) < 1e-6 if (rt_sharpe is not None and alloc_base_sharpe is not None) else False
        cagr_match = abs(rt_cagr - alloc_base_cagr) < 1e-6 if (rt_cagr is not None and alloc_base_cagr is not None) else False
        
        print(f"  Post-RT: Sharpe={rt_sharpe:.4f}, CAGR={rt_cagr:.2%}")
        print(f"  Post-Allocator Base: Sharpe={alloc_base_sharpe:.4f}, CAGR={alloc_base_cagr:.2%}")
        print(f"  Identity check: Sharpe match={sharpe_match}, CAGR match={cagr_match}")
        
        return sharpe_match and cagr_match
    else:
        # Should differ when allocator has effect
        rt_sharpe = post_rt_metrics.get('sharpe')
        alloc_sharpe = post_alloc_metrics.get('sharpe')
        rt_cagr = post_rt_metrics.get('cagr')
        alloc_cagr = post_alloc_metrics.get('cagr')
        
        sharpe_diff = abs(rt_sharpe - alloc_sharpe) if (rt_sharpe is not None and alloc_sharpe is not None) else 0
        cagr_diff = abs(rt_cagr - alloc_cagr) if (rt_cagr is not None and alloc_cagr is not None) else 0
        
        print(f"  Post-RT: Sharpe={rt_sharpe:.4f}, CAGR={rt_cagr:.2%}")
        print(f"  Post-Allocator: Sharpe={alloc_sharpe:.4f}, CAGR={alloc_cagr:.2%}")
        print(f"  Difference: Sharpe diff={sharpe_diff:.4f}, CAGR diff={cagr_diff:.2%}")
        
        # Check if allocator had effect from meta
        meta_path = run_dir / 'meta.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            alloc_meta = meta.get('allocator_v1', {})
            had_clip = alloc_meta.get('had_clip', False)
            print(f"  Allocator had_clip: {had_clip}")
            
            if had_clip:
                # Should differ
                return sharpe_diff > 1e-6 or cagr_diff > 1e-6
            else:
                # May be similar if no clipping occurred
                return True
        
        return True


def main():
    """Verify both baseline runs."""
    print("=" * 80)
    print("Phase 3B Waterfall Stage Semantics Verification")
    print("=" * 80)
    print()
    
    artifacts_only_id = "phase3b_baseline_artifacts_only_20260117_125419"
    traded_id = "phase3b_baseline_traded_20260117_125419"
    
    print("Artifacts-Only Baseline:")
    print("-" * 80)
    artifacts_ok = verify_stage_semantics(artifacts_only_id, is_artifacts_only=True)
    print()
    
    print("Traded Baseline:")
    print("-" * 80)
    traded_ok = verify_stage_semantics(traded_id, is_artifacts_only=False)
    print()
    
    print("=" * 80)
    if artifacts_ok and traded_ok:
        print("[PASS] Stage semantics verified correctly")
        return 0
    else:
        print("[FAIL] Stage semantics verification failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
