import pandas as pd
import asyncio
import httpx
import json
import math
from typing import List
from datetime import datetime

### CONFIGURATION - UPDATE THESE VALUES ###
API_URL = "http://localhost:105/api/classify"
CSV_PATH = "r7_parent_vars.csv"  # Your data file
FATHER_JOB_COLUMN = "fthjobtr7"  # Father's job title column
MOTHER_JOB_COLUMN = "mthjobtr7"  # Mother's job title column
ID_COLUMN = "pidno"  # Person ID column

# API Configuration
BATCH_SIZE = 45  # Number of concurrent requests per batch
TIMEOUT = 50.0  # Timeout in seconds
K_CANDIDATES = 10  # Number of candidate SOC codes to retrieve
INDEX = "soc4d"  # Vector index: "soc4d" or "job-titles-4d"
MODEL = "gpt-4o-mini-2024-07-18"  # OpenAI model to use

# Prompt Configuration
INIT_QUESTION = "What was your main job over the last seven days? Please enter your job title and the industry you are in."
PROMPT_FILE = "../classify_prompt.txt"  # Path to system prompt file

# Missing/Don't Know Response Handling
MISSING_INDICATORS = [
    "don't know", "dont know", "dk", "missing", "na", "n/a",
    "not applicable", "no answer", "refused", "prefer not to say",
    "", "nan", "none", "null"
]
### END CONFIGURATION ###


def is_missing_response(job_title: str) -> bool:
    """
    Check if a job title response is missing or 'don't know'.

    Args:
        job_title: The job title string to check

    Returns:
        True if response is missing/don't know, False otherwise
    """
    if pd.isna(job_title):
        return True

    job_title_lower = str(job_title).lower().strip()
    return job_title_lower in MISSING_INDICATORS


async def classify(client: httpx.AsyncClient, job_title: str, row_id: str = None, job_type: str = ""):
    """
    Send a single job title to the classification API.

    Args:
        client: httpx AsyncClient for making requests
        job_title: The job title/response to classify
        row_id: Optional unique identifier for this response
        job_type: Label for this job (e.g., "father", "mother")

    Returns:
        Dictionary with classification results
    """
    # Check if response is missing or don't know
    if is_missing_response(job_title):
        print(f"⊘ {row_id or 'N/A'} | {job_type} | Missing/Don't Know")
        return {
            "id": row_id,
            "job_type": job_type,
            "job_title": str(job_title) if not pd.isna(job_title) else "NA",
            "soc_code": "NA",
            "soc_description": "NA",
            "confidence": "NA",
            "followup_needed": "NA",
            "candidate_codes": "NA",
            "full_response": "Missing or Don't Know response",
            "skipped": True
        }

    # Read system prompt
    with open(PROMPT_FILE, "r") as f:
        sys_prompt = f.read()

    payload = {
        "sys_prompt": sys_prompt,
        "init_q": INIT_QUESTION,
        "init_ans": job_title,
        "k": K_CANDIDATES,
        "index": INDEX,
        "model": MODEL,
    }

    try:
        response = await client.post(API_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()

        # Log progress
        soc_code = result.get('soc_code', 'N/A')
        confidence = result.get('soc_conf', 'N/A')
        job_title_display = str(job_title)[:40]
        print(f"✓ {row_id or 'N/A'} | {job_type} | {job_title_display} → {soc_code} (conf: {confidence})")

        return {
            "id": row_id,
            "job_type": job_type,
            "job_title": job_title,
            "soc_code": soc_code,
            "soc_description": result.get('soc_desc', 'N/A'),
            "confidence": confidence,
            "followup_needed": result.get('soc_followup', 'N/A'),
            "candidate_codes": result.get('soc_cands', 'N/A'),
            "full_response": result.get('response', 'N/A'),
            "skipped": False
        }
    except Exception as e:
        job_title_display = str(job_title)[:40]
        print(f"❌ {row_id or 'N/A'} | {job_type} | {job_title_display} failed: {e}")
        return {
            "id": row_id,
            "job_type": job_type,
            "job_title": job_title,
            "soc_code": "ERROR",
            "soc_description": "ERROR",
            "confidence": "ERROR",
            "error": str(e),
            "skipped": False
        }


async def process_batch(batch: List[tuple]):
    """
    Process a batch of job titles concurrently.

    Args:
        batch: List of (job_title, row_id, job_type) tuples

    Returns:
        List of classification results
    """
    async with httpx.AsyncClient() as client:
        tasks = [classify(client, title, row_id, job_type) for title, row_id, job_type in batch]
        return await asyncio.gather(*tasks)


async def main():
    """
    Main function to load data, process in batches, and save results.
    """
    print(f"\n{'='*60}")
    print(f"SOC Classification Batch Processing")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading data from: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✓ Loaded {len(df)} records\n")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Validate columns
    missing_cols = []
    if FATHER_JOB_COLUMN not in df.columns:
        missing_cols.append(FATHER_JOB_COLUMN)
    if MOTHER_JOB_COLUMN not in df.columns:
        missing_cols.append(MOTHER_JOB_COLUMN)

    if missing_cols:
        print(f"❌ Error: Columns not found: {', '.join(missing_cols)}")
        print(f"Available columns: {', '.join(df.columns)}")
        return

    # Prepare data - create separate entries for father and mother jobs
    job_data = []
    has_id = ID_COLUMN in df.columns

    for idx, row in df.iterrows():
        row_id = str(row[ID_COLUMN]) if has_id else str(idx)

        # Add father's job
        father_job = row[FATHER_JOB_COLUMN]
        job_data.append((father_job, row_id, "father"))

        # Add mother's job
        mother_job = row[MOTHER_JOB_COLUMN]
        job_data.append((mother_job, row_id, "mother"))

    # Process in batches
    all_results = []
    total_batches = math.ceil(len(job_data) / BATCH_SIZE)

    print(f"Configuration:")
    print(f"  - Model: {MODEL}")
    print(f"  - Index: {INDEX}")
    print(f"  - K candidates: {K_CANDIDATES}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Total records: {len(df)}")
    print(f"  - Total jobs to code: {len(job_data)} (father + mother)")
    print(f"  - Total batches: {total_batches}\n")

    for i in range(total_batches):
        batch = job_data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        print(f"\n{'─'*60}")
        print(f"Batch {i + 1}/{total_batches} ({len(batch)} items)")
        print(f"{'─'*60}")

        batch_results = await process_batch(batch)
        all_results.extend(batch_results)

        # Brief pause between batches to avoid overwhelming the API
        if i < total_batches - 1:
            await asyncio.sleep(2)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./test_results/soc_classification_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary statistics
    total_jobs = len(all_results)
    successful = sum(1 for r in all_results if not r.get('skipped', False) and 'error' not in r)
    skipped = sum(1 for r in all_results if r.get('skipped', False))
    failed = sum(1 for r in all_results if 'error' in r and not r.get('skipped', False))

    father_jobs = [r for r in all_results if r.get('job_type') == 'father']
    mother_jobs = [r for r in all_results if r.get('job_type') == 'mother']

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total jobs processed: {total_jobs}")
    print(f"  - Father jobs: {len(father_jobs)}")
    print(f"  - Mother jobs: {len(mother_jobs)}")
    print(f"\nClassification results:")
    print(f"  - Successfully coded: {successful}")
    print(f"  - Skipped (missing/don't know): {skipped}")
    print(f"  - Failed (errors): {failed}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
