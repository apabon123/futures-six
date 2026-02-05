"""Compare artifacts between two runs to find where divergence starts."""

import pandas as pd
import hashlib
from pathlib import Path

def hash_csv(path):
    df = pd.read_csv(path)
    content = df.to_csv(index=False, float_format='%.15g')
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def main():
    runs = {
        'artifacts_only': 'phase3b_baseline_artifacts_only_20260117_125419',
        'traded': 'phase3b_baseline_traded_20260117_125419',
    }
    
    artifacts = [
        'sleeve_returns.csv',
        'weights_raw.csv',
        'weights_post_construction.csv',
        'weights_post_risk_targeting.csv',
        'weights_post_allocator.csv',
        'asset_returns.csv',
        'portfolio_returns.csv',
    ]
    
    base_dir = Path('reports/runs')
    
    print('Comparing artifacts between artifacts_only and traded runs:')
    print()
    
    for artifact in artifacts:
        hashes = {}
        for name, run_id in runs.items():
            path = base_dir / run_id / artifact
            try:
                hashes[name] = hash_csv(path)
            except Exception as e:
                hashes[name] = f'ERROR: {str(e)[:30]}'
        
        match = hashes['artifacts_only'] == hashes['traded']
        status = 'MATCH' if match else 'DIFF'
        print(f'{artifact}: {status}')
        if not match:
            print(f'  artifacts_only: {hashes["artifacts_only"]}')
            print(f'  traded:         {hashes["traded"]}')


if __name__ == '__main__':
    main()
