"""
Convert SOC classification JSON results to CSV format.

This script takes the JSON output from batch_requests_simple.py and converts it
to a wide-format CSV with one row per respondent and separate columns for
father and mother job classifications.
"""

import json
import pandas as pd
import sys
from pathlib import Path


def json_to_csv(json_file: str, output_csv: str = None):
    """
    Convert JSON results to CSV format.

    Args:
        json_file: Path to the JSON results file
        output_csv: Optional path for output CSV (defaults to same name with .csv)
    """
    # Load JSON results
    with open(json_file, 'r') as f:
        results = json.load(f)

    # Separate father and mother results
    father_results = {r['id']: r for r in results if r.get('job_type') == 'father'}
    mother_results = {r['id']: r for r in results if r.get('job_type') == 'mother'}

    # Get all unique IDs
    all_ids = sorted(set(father_results.keys()) | set(mother_results.keys()))

    # Build output data
    output_data = []
    for rid in all_ids:
        father = father_results.get(rid, {})
        mother = mother_results.get(rid, {})

        output_data.append({
            'id': rid,
            # Father's job
            'father_job_title': father.get('job_title', ''),
            'father_soc_code': father.get('soc_code', ''),
            'father_soc_description': father.get('soc_description', ''),
            'father_confidence': father.get('confidence', ''),
            'father_followup_needed': father.get('followup_needed', ''),
            'father_skipped': father.get('skipped', False),
            # Mother's job
            'mother_job_title': mother.get('job_title', ''),
            'mother_soc_code': mother.get('soc_code', ''),
            'mother_soc_description': mother.get('soc_description', ''),
            'mother_confidence': mother.get('confidence', ''),
            'mother_followup_needed': mother.get('followup_needed', ''),
            'mother_skipped': mother.get('skipped', False),
        })

    # Create DataFrame
    df = pd.DataFrame(output_data)

    # Determine output file
    if output_csv is None:
        json_path = Path(json_file)
        output_csv = str(json_path.with_suffix('.csv'))

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"✓ Converted {len(df)} records to CSV")
    print(f"✓ Saved to: {output_csv}")

    # Print summary statistics
    father_coded = df[df['father_soc_code'].notna() &
                     (df['father_soc_code'] != 'NA') &
                     (df['father_soc_code'] != 'ERROR')].shape[0]
    mother_coded = df[df['mother_soc_code'].notna() &
                     (df['mother_soc_code'] != 'NA') &
                     (df['mother_soc_code'] != 'ERROR')].shape[0]
    father_skipped = df['father_skipped'].sum()
    mother_skipped = df['mother_skipped'].sum()

    print(f"\nSummary:")
    print(f"  Father jobs:")
    print(f"    - Successfully coded: {father_coded}")
    print(f"    - Skipped (missing/don't know): {father_skipped}")
    print(f"  Mother jobs:")
    print(f"    - Successfully coded: {mother_coded}")
    print(f"    - Skipped (missing/don't know): {mother_skipped}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_csv.py <json_file> [output_csv]")
        print("\nExample:")
        print("  python json_to_csv.py test_results/soc_classification_20241111_123456.json")
        sys.exit(1)

    json_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    json_to_csv(json_file, output_csv)
