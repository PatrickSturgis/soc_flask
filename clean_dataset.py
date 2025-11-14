#!/usr/bin/env python3
"""
Clean dataset by removing invalid responses for father's job title.

This script filters out rows where the father's job title is:
- "Don't know"
- Empty/NA/missing
- Special codes (like -9, -8)
- Other invalid responses

The pidno column is preserved to allow merging results back to the original dataset.

Usage:
    python clean_dataset.py --input testing/r7_parent_vars.csv --output testing/r7_cleaned.csv
"""

import argparse
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean dataset by removing invalid job title responses"
    )

    parser.add_argument(
        "--input", "-i",
        default="testing/r7_parent_vars.csv",
        help="Input CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        default="testing/r7_parent_vars_cleaned.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--job-col",
        default="fthjobtr7",
        help="Column name for father's job title"
    )
    parser.add_argument(
        "--id-col",
        default="pidno",
        help="Column name for unique identifier (for merging back later)"
    )
    parser.add_argument(
        "--keep-mother",
        action="store_true",
        help="Also process mother's job title (mthjobtr7)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics about removed cases"
    )

    return parser.parse_args()


def is_valid_job_title(value):
    """
    Check if a job title is valid.

    Returns True if the job title should be kept, False if it should be filtered out.
    """
    # Convert to string and strip whitespace
    if pd.isna(value):
        return False

    value_str = str(value).strip()

    # Empty string
    if not value_str:
        return False

    # Don't know variations
    dont_know_patterns = [
        "don't know",
        "dont know",
        "do not know",
        "dk",
        "unknown",
    ]

    if value_str.lower() in dont_know_patterns:
        return False

    # Special codes (negative numbers, -9, -8, etc.)
    try:
        num_value = float(value_str)
        if num_value < 0:
            return False
    except ValueError:
        # Not a number, continue checking
        pass

    # Refusal patterns
    refusal_patterns = [
        "refused",
        "refusal",
        "prefer not to say",
        "n/a",
        "na",
    ]

    if value_str.lower() in refusal_patterns:
        return False

    # If we get here, it's probably a valid job title
    return True


def analyze_removed_cases(df, job_col, mask):
    """Analyze which cases are being removed and why."""
    removed_df = df[~mask]

    logger.info("="*80)
    logger.info("Analysis of Removed Cases")
    logger.info("="*80)

    # Count by value
    value_counts = removed_df[job_col].value_counts(dropna=False)

    logger.info(f"\nTop reasons for removal (by frequency):")
    for i, (value, count) in enumerate(value_counts.head(20).items()):
        if pd.isna(value):
            logger.info(f"  {i+1}. Missing/NA: {count:,}")
        else:
            logger.info(f"  {i+1}. '{value}': {count:,}")

    # Check for other columns that might be correlated
    if 'dvager7' in removed_df.columns:
        logger.info(f"\nAge distribution of removed cases:")
        age_valid = removed_df['dvager7'][removed_df['dvager7'] > 0]
        if len(age_valid) > 0:
            age_stats = age_valid.describe()
            logger.info(f"  Mean age: {age_stats['mean']:.1f}")
            logger.info(f"  Min age: {age_stats['min']:.1f}")
            logger.info(f"  Max age: {age_stats['max']:.1f}")

    logger.info("="*80)


def main():
    args = parse_args()

    logger.info("="*80)
    logger.info("Dataset Cleaning Tool")
    logger.info("="*80)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Job column: {args.job_col}")
    logger.info(f"ID column: {args.id_col}")
    logger.info("="*80)

    # Load data
    logger.info(f"\nLoading data from {args.input}...")
    df = pd.read_csv(args.input)

    original_count = len(df)
    logger.info(f"Original dataset: {original_count:,} rows")

    # Check if columns exist
    if args.job_col not in df.columns:
        logger.error(f"Column '{args.job_col}' not found in dataset")
        logger.error(f"Available columns: {list(df.columns)}")
        return

    if args.id_col not in df.columns:
        logger.warning(f"ID column '{args.id_col}' not found in dataset")
        logger.warning(f"Creating sequential IDs...")
        df[args.id_col] = range(len(df))

    # Check for duplicate IDs
    duplicate_ids = df[args.id_col].duplicated().sum()
    if duplicate_ids > 0:
        logger.warning(f"⚠ Found {duplicate_ids} duplicate IDs in '{args.id_col}'")
        logger.warning(f"This may cause issues when merging results back")

    # Show sample of data
    logger.info(f"\nSample of '{args.job_col}' column:")
    logger.info(df[args.job_col].value_counts(dropna=False).head(10))

    # Filter valid job titles
    logger.info(f"\nFiltering invalid job titles...")
    valid_mask = df[args.job_col].apply(is_valid_job_title)

    # If processing mother's job title too
    if args.keep_mother and 'mthjobtr7' in df.columns:
        logger.info("Also filtering mother's job title...")
        valid_mask_mother = df['mthjobtr7'].apply(is_valid_job_title)
        # Keep rows where EITHER father OR mother has valid job title
        valid_mask = valid_mask | valid_mask_mother

    filtered_df = df[valid_mask].copy()
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count

    # Statistics
    logger.info("\n" + "="*80)
    logger.info("Filtering Results")
    logger.info("="*80)
    logger.info(f"Original rows: {original_count:,}")
    logger.info(f"Valid rows: {filtered_count:,}")
    logger.info(f"Removed rows: {removed_count:,}")
    logger.info(f"Retention rate: {(filtered_count/original_count)*100:.1f}%")
    logger.info(f"Reduction: {(removed_count/original_count)*100:.1f}%")
    logger.info("="*80)

    # Detailed analysis if requested
    if args.stats:
        analyze_removed_cases(df, args.job_col, valid_mask)

    # Save filtered data
    logger.info(f"\nSaving filtered data to {args.output}...")

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_df.to_csv(args.output, index=False)

    logger.info(f"✓ Saved {filtered_count:,} rows to {args.output}")

    # Show sample of cleaned data
    logger.info(f"\nSample of cleaned '{args.job_col}' column:")
    sample_counts = filtered_df[args.job_col].value_counts()
    for i, (value, count) in enumerate(sample_counts.head(10).items()):
        logger.info(f"  {i+1}. '{value}': {count:,}")

    # File size comparison
    original_size = Path(args.input).stat().st_size / (1024*1024)  # MB
    filtered_size = output_path.stat().st_size / (1024*1024)  # MB

    logger.info("\n" + "="*80)
    logger.info("File Size Comparison")
    logger.info("="*80)
    logger.info(f"Original: {original_size:.2f} MB")
    logger.info(f"Cleaned: {filtered_size:.2f} MB")
    logger.info(f"Reduction: {((original_size - filtered_size) / original_size * 100):.1f}%")
    logger.info("="*80)

    # Instructions for merging back
    logger.info("\n" + "="*80)
    logger.info("Merging Results Back Later")
    logger.info("="*80)
    logger.info(f"After running SOC classification on the cleaned data, merge back using:")
    logger.info(f"")
    logger.info(f"  import pandas as pd")
    logger.info(f"  ")
    logger.info(f"  # Load original and results")
    logger.info(f"  original = pd.read_csv('{args.input}')")
    logger.info(f"  results = pd.read_csv('soc_results.csv')")
    logger.info(f"  ")
    logger.info(f"  # Merge back (left join keeps all original rows)")
    logger.info(f"  final = original.merge(")
    logger.info(f"      results[['{args.id_col}', 'soc_code', 'soc_desc', 'soc_conf']],")
    logger.info(f"      on='{args.id_col}',")
    logger.info(f"      how='left'")
    logger.info(f"  )")
    logger.info(f"  ")
    logger.info(f"  # Rows without valid job titles will have NaN for SOC codes")
    logger.info(f"  final.to_csv('final_with_soc_codes.csv', index=False)")
    logger.info("="*80)

    logger.info("\n✓ Cleaning complete!")


if __name__ == "__main__":
    main()
