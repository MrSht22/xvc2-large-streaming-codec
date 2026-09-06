#!/usr/bin/env python3
"""Fix ja dataset CSV to include subset information (nonpara30, parallel100, both)."""

import pandas as pd
import sys
from pathlib import Path

def fix_ja_csv(csv_path: Path, output_path: Path | None = None):
    """Fix ja dataset entries to include subset information.
    
    Args:
        csv_path: Path to the CSV file to fix
        output_path: Optional output path. If None, overwrites the original file.
    """
    df = pd.read_csv(csv_path)
    
    # Find all ja entries
    ja_mask = df['dataset'] == 'ja'
    
    if not ja_mask.any():
        print("No 'ja' entries found in CSV file.")
        return
    
    print(f"Found {ja_mask.sum()} 'ja' entries.")
    print("\nPlease specify which rows correspond to which subset:")
    print("  - nonpara30: rows for ja_dev_nonpara30 and ja_test_nonpara30")
    print("  - parallel100: rows for ja_dev_parallel100 and ja_test_parallel100")
    print("  - both: rows for ja_dev_both and ja_test_both")
    print("\nEach subset should have 12 rows (3 scenarios × 2 splits × 2 genders)")
    
    # Get ja row indices
    ja_indices = df[ja_mask].index.tolist()
    print(f"\nJA row indices: {ja_indices}")
    
    # Ask user for mapping
    print("\nEnter row ranges for each subset (format: start-end, e.g., 60-71):")
    nonpara30_range = input("nonpara30 rows (start-end): ").strip()
    parallel100_range = input("parallel100 rows (start-end): ").strip()
    both_range = input("both rows (start-end, or press Enter if none): ").strip()
    
    def parse_range(range_str):
        if not range_str:
            return []
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    
    nonpara30_rows = parse_range(nonpara30_range)
    parallel100_rows = parse_range(parallel100_range)
    both_rows = parse_range(both_range) if both_range else []
    
    # Update dataset names
    for idx in nonpara30_rows:
        if idx in df.index and df.loc[idx, 'dataset'] == 'ja':
            df.loc[idx, 'dataset'] = 'ja_nonpara30'
    
    for idx in parallel100_rows:
        if idx in df.index and df.loc[idx, 'dataset'] == 'ja':
            df.loc[idx, 'dataset'] = 'ja_parallel100'
    
    for idx in both_rows:
        if idx in df.index and df.loc[idx, 'dataset'] == 'ja':
            df.loc[idx, 'dataset'] = 'ja_both'
    
    # Save
    output = output_path or csv_path
    df.to_csv(output, index=False)
    print(f"\nFixed CSV saved to: {output}")
    print(f"\nUpdated dataset names:")
    print(f"  ja_nonpara30: {len(nonpara30_rows)} rows")
    print(f"  ja_parallel100: {len(parallel100_rows)} rows")
    if both_rows:
        print(f"  ja_both: {len(both_rows)} rows")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_ja_csv.py <csv_file> [output_file]")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    fix_ja_csv(csv_path, output_path)
