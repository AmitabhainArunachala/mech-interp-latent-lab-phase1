#!/usr/bin/env python3
"""
OpenClaw Quick-Start Implementation
Proof-of-concept for Pipeline 1: Experiment Results Consolidation

This script demonstrates read-only data aggregation as a safe first step.
NO MODIFICATIONS to source data - only reads and creates new consolidated files.

Usage:
    python openclaw_quickstart.py --dry-run        # Preview only
    python openclaw_quickstart.py --validate       # Run and validate
    python openclaw_quickstart.py --output results/consolidated/

Requirements:
    pip install pandas
"""

import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    sys.exit(1)


class ExperimentAggregator:
    """Read-only aggregation of experiment results from RUN_INDEX.jsonl"""

    def __init__(self, base_dir: Path, dry_run: bool = False):
        self.base_dir = base_dir
        self.dry_run = dry_run
        self.audit_log: List[Dict] = []

    def log_operation(self, operation: str, file_path: str, success: bool,
                     details: Optional[str] = None):
        """Log all operations for audit trail"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "file_path": file_path,
            "success": success,
            "details": details,
            "dry_run": self.dry_run
        }
        self.audit_log.append(entry)

        # Print to console
        status = "✓" if success else "✗"
        print(f"[{status}] {operation}: {file_path}")
        if details:
            print(f"    {details}")

    def checksum_file(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(65536), b''):
                sha256.update(block)
        return sha256.hexdigest()

    def validate_schema(self, entry: Dict) -> tuple[bool, Optional[str]]:
        """Validate JSONL entry against expected schema"""
        required_fields = ['timestamp', 'experiment', 'success']

        # Check required fields
        missing = [f for f in required_fields if f not in entry]
        if missing:
            return False, f"Missing required fields: {missing}"

        # Validate types
        if not isinstance(entry['success'], bool):
            return False, "Field 'success' must be boolean"

        # Validate optional numeric fields
        numeric_fields = ['rv_d', 'rv_p', 'rv_delta', 'logit_diff_d',
                         'logit_diff_p', 'n_pairs']
        for field in numeric_fields:
            if field in entry and entry[field] is not None:
                try:
                    float(entry[field])
                except (ValueError, TypeError):
                    return False, f"Field '{field}' must be numeric or null"

        return True, None

    def load_run_index(self) -> pd.DataFrame:
        """Load and validate RUN_INDEX.jsonl"""
        run_index_path = self.base_dir / "results" / "RUN_INDEX.jsonl"

        if not run_index_path.exists():
            self.log_operation("load_run_index", str(run_index_path),
                             False, "File not found")
            return pd.DataFrame()

        # Compute checksum for integrity
        checksum = self.checksum_file(run_index_path)

        # Read JSONL
        entries = []
        anomalies = []

        with open(run_index_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)

                    # Validate schema
                    valid, error = self.validate_schema(entry)
                    if not valid:
                        anomalies.append({
                            'line': line_num,
                            'error': error,
                            'entry': entry
                        })
                        continue

                    entries.append(entry)

                except json.JSONDecodeError as e:
                    anomalies.append({
                        'line': line_num,
                        'error': f"JSON decode error: {e}",
                        'entry': line[:100]
                    })

        self.log_operation("load_run_index", str(run_index_path), True,
                         f"Loaded {len(entries)} entries, {len(anomalies)} anomalies")

        # Store anomalies for reporting
        self.anomalies = anomalies

        return pd.DataFrame(entries)

    def compute_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute aggregate statistics"""
        stats = {
            'total_runs': len(df),
            'successful_runs': df['success'].sum(),
            'success_rate': df['success'].mean() * 100,
            'unique_experiments': df['experiment'].nunique(),
            'unique_models': df['model'].nunique() if 'model' in df.columns else 0,
        }

        # Experiments by type
        stats['runs_by_experiment'] = df['experiment'].value_counts().to_dict()

        # Success rate by experiment
        if len(df) > 0:
            stats['success_by_experiment'] = df.groupby('experiment')['success'].mean().to_dict()

        # R_V statistics (only successful runs with R_V data)
        rv_data = df[df['success'] & df['rv_delta'].notna()]
        if len(rv_data) > 0:
            stats['rv_statistics'] = {
                'count': len(rv_data),
                'mean_delta': float(rv_data['rv_delta'].mean()),
                'std_delta': float(rv_data['rv_delta'].std()),
                'min_delta': float(rv_data['rv_delta'].min()),
                'max_delta': float(rv_data['rv_delta'].max()),
            }

            # Significant results (p < 0.001)
            if 'rv_p' in rv_data.columns:
                sig_results = rv_data[rv_data['rv_p'] < 0.001]
                stats['rv_statistics']['significant_results'] = len(sig_results)
                stats['rv_statistics']['significance_rate'] = len(sig_results) / len(rv_data) * 100

        # By model (if available)
        if 'model' in df.columns and len(rv_data) > 0:
            stats['rv_by_model'] = rv_data.groupby('model')['rv_delta'].agg(['count', 'mean', 'std']).to_dict()

        return stats

    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect unusual patterns in data"""
        anomaly_list = list(self.anomalies)  # Start with schema anomalies

        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            dupes = df[df.duplicated(subset=['timestamp'], keep=False)]
            if len(dupes) > 0:
                anomaly_list.append({
                    'type': 'duplicate_timestamp',
                    'count': len(dupes),
                    'examples': dupes['timestamp'].tolist()[:5]
                })

        # Check for R_V delta out of expected range
        if 'rv_delta' in df.columns:
            out_of_range = df[(df['rv_delta'] < -2.0) | (df['rv_delta'] > 2.0)]
            if len(out_of_range) > 0:
                anomaly_list.append({
                    'type': 'rv_delta_out_of_range',
                    'count': len(out_of_range),
                    'examples': out_of_range[['timestamp', 'experiment', 'rv_delta']].head().to_dict('records')
                })

        # Check for missing critical fields in successful runs
        successful = df[df['success'] == True]
        if len(successful) > 0:
            for field in ['rv_d', 'rv_p', 'rv_delta']:
                missing = successful[successful[field].isna()]
                if len(missing) > 0:
                    anomaly_list.append({
                        'type': f'missing_{field}_in_successful_run',
                        'count': len(missing),
                        'examples': missing[['timestamp', 'experiment']].head().to_dict('records')
                    })

        return anomaly_list

    def save_outputs(self, df: pd.DataFrame, stats: Dict, anomalies: List[Dict],
                    output_dir: Path):
        """Save consolidated results (with dry-run support)"""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Main consolidated CSV
        csv_path = output_dir / "all_runs_summary.csv"
        if self.dry_run:
            print(f"\n[DRY RUN] Would write {len(df)} rows to: {csv_path}")
            print(f"Columns: {list(df.columns)}")
        else:
            df.to_csv(csv_path, index=False)
            self.log_operation("write_csv", str(csv_path), True,
                             f"Wrote {len(df)} rows")

        # 2. Summary statistics JSON
        stats_path = output_dir / "summary_statistics.json"
        if self.dry_run:
            print(f"\n[DRY RUN] Would write stats to: {stats_path}")
            print(json.dumps(stats, indent=2)[:500] + "...")
        else:
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            self.log_operation("write_json", str(stats_path), True)

        # 3. Anomalies log
        anomaly_path = output_dir / f"anomalies_{timestamp}.log"
        if len(anomalies) > 0:
            if self.dry_run:
                print(f"\n[DRY RUN] Would write {len(anomalies)} anomalies to: {anomaly_path}")
            else:
                with open(anomaly_path, 'w') as f:
                    f.write(f"Anomaly Report - {timestamp}\n")
                    f.write("=" * 80 + "\n\n")
                    for i, anomaly in enumerate(anomalies, 1):
                        f.write(f"{i}. {json.dumps(anomaly, indent=2)}\n\n")
                self.log_operation("write_log", str(anomaly_path), True,
                                 f"Logged {len(anomalies)} anomalies")

        # 4. Last run timestamp
        lastrun_path = output_dir / "last_run.txt"
        if not self.dry_run:
            with open(lastrun_path, 'w') as f:
                f.write(timestamp)
            self.log_operation("write_timestamp", str(lastrun_path), True)

    def save_audit_log(self, output_dir: Path):
        """Save audit log of all operations"""
        if self.dry_run:
            print(f"\n[DRY RUN] Would write {len(self.audit_log)} audit entries")
            return

        audit_path = output_dir / "audit_log.jsonl"
        with open(audit_path, 'a') as f:  # Append mode
            for entry in self.audit_log:
                f.write(json.dumps(entry) + '\n')

        print(f"\n✓ Audit log saved: {audit_path} ({len(self.audit_log)} entries)")

    def run(self, output_dir: Path) -> bool:
        """Main execution pipeline"""
        print("=" * 80)
        print("OpenClaw Pipeline: Experiment Results Consolidation")
        print("=" * 80)
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Dry run: {self.dry_run}")
        print("=" * 80 + "\n")

        # Step 1: Load RUN_INDEX.jsonl
        print("Step 1: Loading RUN_INDEX.jsonl...")
        df = self.load_run_index()
        if df.empty:
            print("ERROR: No data loaded. Aborting.")
            return False
        print(f"✓ Loaded {len(df)} experiment runs\n")

        # Step 2: Compute summary statistics
        print("Step 2: Computing summary statistics...")
        stats = self.compute_summary_stats(df)
        print(f"✓ Total runs: {stats['total_runs']}")
        print(f"✓ Success rate: {stats['success_rate']:.1f}%")
        if 'rv_statistics' in stats:
            print(f"✓ R_V measurements: {stats['rv_statistics']['count']}")
            print(f"✓ Mean R_V delta: {stats['rv_statistics']['mean_delta']:.4f}\n")

        # Step 3: Detect anomalies
        print("Step 3: Detecting anomalies...")
        anomalies = self.detect_anomalies(df)
        print(f"✓ Anomalies detected: {len(anomalies)}\n")

        # Step 4: Save outputs
        print("Step 4: Saving outputs...")
        self.save_outputs(df, stats, anomalies, output_dir)

        # Step 5: Save audit log
        self.save_audit_log(output_dir)

        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)

        return True


def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw Quick-Start: Aggregate experiment results"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.home() / "mech-interp-latent-lab-phase1",
        help="Base directory of research repo"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: <base-dir>/results/consolidated/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing files"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks and report statistics"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = args.base_dir / "results" / "consolidated"

    # Validate base directory
    if not args.base_dir.exists():
        print(f"ERROR: Base directory does not exist: {args.base_dir}")
        sys.exit(1)

    # Run aggregator
    aggregator = ExperimentAggregator(args.base_dir, dry_run=args.dry_run)
    success = aggregator.run(output_dir)

    if success:
        print("\nNext steps:")
        print("1. Review outputs in:", output_dir)
        print("2. Validate against known ground truth (e.g., PHASE1_FINAL_REPORT.md)")
        print("3. If accurate, disable --dry-run and run again")
        print("4. Set up daily cron job for automated aggregation")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
