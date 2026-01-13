"""
System Cleanup Audit - Step 3
==============================

This script identifies non-functional issues for cleanup:
- Logging gaps
- Artifact naming inconsistencies
- Missing meta fields
- Run reproducibility sharp edges
- Confusing config boundaries
- Ambiguous defaults
- "This shouldn't be possible" guardrails

This is NOT tuning or adding logic. This is strictly non-functional cleanup.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import json
import yaml
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CleanupAudit:
    """Audit system for non-functional issues."""
    
    def __init__(self):
        self.issues = []
    
    def add_issue(self, category: str, severity: str, description: str, file: str = None, line: int = None, fix_suggestion: str = None):
        """Add an issue to the audit."""
        self.issues.append({
            'category': category,
            'severity': severity,  # 'low', 'medium', 'high'
            'description': description,
            'file': file,
            'line': line,
            'fix_suggestion': fix_suggestion
        })
    
    def audit_logging_gaps(self):
        """Check for logging gaps."""
        logger.info("Auditing logging gaps...")
        
        # Check if run_strategy.py logs effective start date
        run_strategy_file = project_root / "run_strategy.py"
        if run_strategy_file.exists():
            content = run_strategy_file.read_text(encoding='utf-8', errors='ignore')
            if "effective start" not in content.lower() and "effective_start" not in content:
                self.add_issue(
                    category="Logging",
                    severity="medium",
                    description="run_strategy.py does not log effective start date (after warmup)",
                    file="run_strategy.py",
                    fix_suggestion="Add logging of effective start date per PROCEDURES.md § 2.3"
                )
        
        # Check if ExecSim logs config settings
        exec_sim_file = project_root / "src/agents/exec_sim.py"
        if exec_sim_file.exists():
            content = exec_sim_file.read_text(encoding='utf-8', errors='ignore')
            if "allocator_v1.enabled" in content and "allocator_v1.mode" in content:
                # Good - config is logged
                pass
            else:
                self.add_issue(
                    category="Logging",
                    severity="low",
                    description="ExecSim may not log all critical config settings",
                    file="src/agents/exec_sim.py"
                )
    
    def audit_naming_consistency(self):
        """Check for naming inconsistencies."""
        logger.info("Auditing naming consistency...")
        
        # Check artifact file naming
        artifact_patterns = {
            'allocator': ['allocator_risk_v1_applied.csv', 'allocator_regime_v1.csv', 'allocator_state_v1.csv'],
            'engine_policy': ['engine_policy_applied_v1.csv', 'engine_policy_state_v1.csv'],
            'risk_targeting': ['leverage_series.csv', 'realized_vol.csv']
        }
        
        # Check for inconsistent naming patterns
        exec_sim_file = project_root / "src/agents/exec_sim.py"
        if exec_sim_file.exists():
            content = exec_sim_file.read_text(encoding='utf-8', errors='ignore')
            # Look for artifact file references
            if 'allocator_risk_v1_applied.csv' in content and 'allocator_regime_v1.csv' in content:
                # Check if naming is consistent
                if 'allocator_risk_v1.csv' in content and 'allocator_risk_v1_applied.csv' in content:
                    # Good - consistent pattern
                    pass
                else:
                    self.add_issue(
                        category="Naming",
                        severity="low",
                        description="Potential inconsistency in allocator artifact naming",
                        file="src/agents/exec_sim.py"
                    )
    
    def audit_meta_fields(self):
        """Check for missing meta fields."""
        logger.info("Auditing meta fields...")
        
        # Check if meta.json includes all required fields
        # This would require checking actual run outputs, so we'll document the expected structure
        expected_meta_fields = [
            'run_id',
            'start_date',
            'end_date',
            'strategy_profile',
            'config_hash',  # For reproducibility
            'canonical_window'  # Whether this uses canonical window
        ]
        
        self.add_issue(
            category="Meta Fields",
            severity="medium",
            description="Verify meta.json includes all required fields for reproducibility",
            fix_suggestion="Ensure meta.json includes: run_id, start_date, end_date, strategy_profile, config_hash, canonical_window"
        )
    
    def audit_reproducibility(self):
        """Check for reproducibility issues."""
        logger.info("Auditing reproducibility...")
        
        # Check if config files have deterministic ordering
        strategies_file = project_root / "configs/strategies.yaml"
        if strategies_file.exists():
            # Check for any non-deterministic patterns (this is hard to detect automatically)
            self.add_issue(
                category="Reproducibility",
                severity="low",
                description="Verify YAML config files maintain deterministic ordering",
                file="configs/strategies.yaml"
            )
        
        # Check if artifact writer uses deterministic sorting
        artifact_writer_file = project_root / "src/layers/artifact_writer.py"
        if artifact_writer_file.exists():
            content = artifact_writer_file.read_text(encoding='utf-8', errors='ignore')
            if 'sort_values' in content or 'sort_index' in content:
                # Good - sorting is applied
                pass
            else:
                self.add_issue(
                    category="Reproducibility",
                    severity="medium",
                    description="ArtifactWriter may not sort data deterministically",
                    file="src/layers/artifact_writer.py",
                    fix_suggestion="Ensure all DataFrame outputs are sorted by date and instrument for reproducibility"
                )
    
    def audit_config_boundaries(self):
        """Check for confusing config boundaries."""
        logger.info("Auditing config boundaries...")
        
        strategies_file = project_root / "configs/strategies.yaml"
        if strategies_file.exists():
            try:
                with open(strategies_file, 'r', encoding='utf-8', errors='ignore') as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not load strategies.yaml: {e}")
                config = {}
            
            # Check for ambiguous defaults
            allocator_config = config.get('allocator_v1', {})
            if allocator_config.get('mode') == 'precomputed' and not allocator_config.get('precomputed_run_id'):
                self.add_issue(
                    category="Config Boundaries",
                    severity="high",
                    description="allocator_v1.mode='precomputed' but precomputed_run_id is null - this will default to 'off' mode",
                    file="configs/strategies.yaml",
                    fix_suggestion="Either set mode='off' or provide a default precomputed_run_id, or add validation"
                )
            
            engine_policy_config = config.get('engine_policy_v1', {})
            if engine_policy_config.get('mode') == 'precomputed' and not engine_policy_config.get('precomputed_run_id'):
                self.add_issue(
                    category="Config Boundaries",
                    severity="high",
                    description="engine_policy_v1.mode='precomputed' but precomputed_run_id is null",
                    file="configs/strategies.yaml",
                    fix_suggestion="Either set mode='off' or provide validation for precomputed mode"
                )
    
    def audit_guardrails(self):
        """Check for missing guardrails."""
        logger.info("Auditing guardrails...")
        
        # Check if there are guardrails against impossible states
        exec_sim_file = project_root / "src/agents/exec_sim.py"
        if exec_sim_file.exists():
            content = exec_sim_file.read_text(encoding='utf-8', errors='ignore')
            
            # Check for validation of precomputed mode requirements
            if 'precomputed' in content and 'precomputed_run_id' in content:
                # Check if there's validation
                if 'if.*precomputed.*and.*not.*precomputed_run_id' in content or 'ValueError' in content:
                    # Good - validation exists
                    pass
                else:
                    self.add_issue(
                        category="Guardrails",
                        severity="medium",
                        description="May be missing validation for precomputed mode requirements",
                        file="src/agents/exec_sim.py",
                        fix_suggestion="Add explicit validation that precomputed mode requires precomputed_run_id"
                    )
    
    def audit_defaults(self):
        """Check for ambiguous defaults."""
        logger.info("Auditing defaults...")
        
        strategies_file = project_root / "configs/strategies.yaml"
        if strategies_file.exists():
            try:
                with open(strategies_file, 'r', encoding='utf-8', errors='ignore') as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not load strategies.yaml: {e}")
                config = {}
            
            # Check for defaults that might be ambiguous
            risk_targeting = config.get('risk_targeting', {})
            if risk_targeting.get('enabled') is None:
                self.add_issue(
                    category="Defaults",
                    severity="low",
                    description="risk_targeting.enabled default is not explicit",
                    file="configs/strategies.yaml"
                )
    
    def run_all_audits(self):
        """Run all audit checks."""
        logger.info("=" * 80)
        logger.info("SYSTEM CLEANUP AUDIT - Step 3")
        logger.info("=" * 80)
        
        self.audit_logging_gaps()
        self.audit_naming_consistency()
        self.audit_meta_fields()
        self.audit_reproducibility()
        self.audit_config_boundaries()
        self.audit_guardrails()
        self.audit_defaults()
        
        return self.issues
    
    def generate_report(self, output_path: Path = None) -> Dict:
        """Generate audit report."""
        if output_path is None:
            output_path = project_root / "reports" / "system_cleanup_audit.json"
        
        # Group issues by category and severity
        report = {
            'generated_at': str(Path(__file__).stat().st_mtime),
            'total_issues': len(self.issues),
            'by_category': {},
            'by_severity': {
                'high': [],
                'medium': [],
                'low': []
            },
            'issues': self.issues
        }
        
        # Group by category
        for issue in self.issues:
            cat = issue['category']
            if cat not in report['by_category']:
                report['by_category'][cat] = []
            report['by_category'][cat].append(issue)
            
            # Group by severity
            sev = issue['severity']
            report['by_severity'][sev].append(issue)
        
        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="System Cleanup Audit - Step 3",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for audit report (default: reports/system_cleanup_audit.json)'
    )
    
    args = parser.parse_args()
    
    audit = CleanupAudit()
    issues = audit.run_all_audits()
    
    output_path = Path(args.output) if args.output else None
    report = audit.generate_report(output_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SYSTEM CLEANUP AUDIT SUMMARY")
    print("=" * 80)
    print(f"\nTotal Issues Found: {len(issues)}")
    print(f"\nBy Severity:")
    print(f"  High:   {len(report['by_severity']['high'])}")
    print(f"  Medium: {len(report['by_severity']['medium'])}")
    print(f"  Low:    {len(report['by_severity']['low'])}")
    
    print(f"\nBy Category:")
    for category, cat_issues in report['by_category'].items():
        print(f"  {category}: {len(cat_issues)}")
    
    if report['by_severity']['high']:
        print("\nHIGH SEVERITY ISSUES:")
        for issue in report['by_severity']['high']:
            print(f"  - {issue['description']}")
            if issue.get('file'):
                print(f"    File: {issue['file']}")
            if issue.get('fix_suggestion'):
                print(f"    Fix: {issue['fix_suggestion']}")
    
    print("\n" + "=" * 80)
    if output_path:
        print(f"Full report saved to: {output_path}")
    print("=" * 80)
    
    return 0 if len(report['by_severity']['high']) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

