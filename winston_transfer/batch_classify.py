#!/usr/bin/env python3
"""
Batch SOC20 Classification Script

Process a CSV file of survey responses and classify job titles into SOC codes.
Designed to run on Winston cluster via Slurm.

Usage:
    python batch_classify.py --input survey_responses.csv --output results.csv --model qwen-7b
"""

import argparse
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

from soc_classifier_local import SOCClassifier
from config import MODELS, EMBEDDING_MODELS, DEFAULT_PARAMS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_classify.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch SOC20 classification for survey responses"
    )

    # Input/output
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file with survey responses"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file for results"
    )

    # Model selection
    parser.add_argument(
        "--chat-model", "-m",
        default="qwen-7b",
        choices=list(MODELS.keys()),
        help=f"Chat model to use. Options: {', '.join(MODELS.keys())}"
    )
    parser.add_argument(
        "--embedding-model", "-e",
        default="all-mpnet-base-v2",
        choices=list(EMBEDDING_MODELS.keys()),
        help=f"Embedding model to use. Options: {', '.join(EMBEDDING_MODELS.keys())}"
    )

    # Column names in input CSV
    parser.add_argument(
        "--question-col",
        default="question",
        help="Column name for the question asked"
    )
    parser.add_argument(
        "--answer-col",
        default="answer",
        help="Column name for the respondent's answer (job title)"
    )
    parser.add_argument(
        "--id-col",
        default="id",
        help="Column name for respondent ID (optional)"
    )

    # Processing parameters
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_PARAMS["k"],
        help="Number of candidate SOC codes to retrieve"
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_PARAMS["collection_name"],
        help="ChromaDB collection to query"
    )
    parser.add_argument(
        "--prompt",
        default="force_classify.txt",
        help="Prompt file to use (from static/ directory)"
    )
    parser.add_argument(
        "--mode",
        choices=["classify", "followup"],
        default="classify",
        help="Classification mode: 'classify' (forced) or 'followup' (interactive)"
    )
    parser.add_argument(
        "--max-followups",
        type=int,
        default=3,
        help="Maximum number of follow-up questions in followup mode"
    )

    # Checkpoint and resume
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N rows"
    )
    parser.add_argument(
        "--resume-from",
        help="Resume from checkpoint file"
    )

    # Other options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Process N rows at a time (not yet implemented)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit processing to first N rows (for testing)"
    )

    return parser.parse_args()


def load_data(input_file: str, id_col: str, question_col: str, answer_col: str) -> pd.DataFrame:
    """Load input CSV file."""
    logger.info(f"Loading data from {input_file}")

    df = pd.read_csv(input_file)

    # Check required columns exist
    required_cols = [question_col, answer_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Add ID column if not present
    if id_col not in df.columns:
        logger.warning(f"ID column '{id_col}' not found, creating sequential IDs")
        df[id_col] = range(len(df))

    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def process_row_classify(
    classifier: SOCClassifier,
    question: str,
    answer: str,
    k: int,
    collection: str,
    prompt: str,
) -> dict:
    """Process a single row in classify mode."""
    try:
        result = classifier.classify(
            init_q=question,
            init_ans=answer,
            sys_prompt=prompt,
            k=k,
            collection_name=collection,
        )
        result["error"] = None
        return result

    except Exception as e:
        logger.error(f"Error processing row: {e}")
        return {
            "soc_code": "ERROR",
            "soc_desc": "ERROR",
            "soc_conf": "ERROR",
            "soc_followup": "ERROR",
            "soc_cands": "",
            "response": "",
            "error": str(e),
        }


def process_row_followup(
    classifier: SOCClassifier,
    question: str,
    answer: str,
    k: int,
    collection: str,
    prompt: str,
    max_followups: int,
) -> dict:
    """Process a single row in followup mode (interactive classification)."""
    additional_qs = []
    followup_questions = []
    result = None

    try:
        for i in range(max_followups + 1):
            result = classifier.followup(
                init_q=question,
                init_ans=answer,
                sys_prompt=prompt,
                k=k,
                collection_name=collection,
                additional_qs=additional_qs if additional_qs else None,
            )

            # Check if we got a code
            if result["soc_code"] not in ["NONE", "ERROR"]:
                logger.info(f"Got SOC code after {i} follow-ups: {result['soc_code']}")
                break

            # Check if we got a follow-up question
            if result["soc_code"] == "NONE" and i < max_followups:
                followup_q = result["followup"]
                followup_questions.append(followup_q)
                logger.info(f"Follow-up {i+1}: {followup_q}")

                # In batch mode, we can't get user input, so we break
                # In interactive mode, you'd prompt the user here
                logger.warning("Cannot answer follow-up in batch mode, using final prompt")
                break

        # Add follow-up info to result
        result["followup_questions"] = json.dumps(followup_questions)
        result["num_followups"] = len(followup_questions)
        result["error"] = None

        return result

    except Exception as e:
        logger.error(f"Error processing row: {e}")
        return {
            "soc_code": "ERROR",
            "soc_desc": "ERROR",
            "soc_conf": "ERROR",
            "followup": "",
            "soc_cands": "",
            "followup_questions": "[]",
            "num_followups": 0,
            "error": str(e),
        }


def save_checkpoint(df: pd.DataFrame, checkpoint_file: str):
    """Save checkpoint."""
    df.to_csv(checkpoint_file, index=False)
    logger.info(f"Checkpoint saved to {checkpoint_file}")


def main():
    args = parse_args()

    # Log configuration
    logger.info("="*80)
    logger.info("SOC20 Batch Classification")
    logger.info("="*80)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Chat model: {args.chat_model} ({MODELS[args.chat_model]['description']})")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"k: {args.k}")
    logger.info(f"Collection: {args.collection}")
    logger.info("="*80)

    # Load data
    df = load_data(args.input, args.id_col, args.question_col, args.answer_col)

    # Apply limit if specified
    if args.limit:
        logger.info(f"Limiting to first {args.limit} rows")
        df = df.head(args.limit)

    # Resume from checkpoint if specified
    start_idx = 0
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_df = pd.read_csv(args.resume_from)
        start_idx = len(checkpoint_df)
        logger.info(f"Resuming from row {start_idx}")

    # Initialize classifier
    logger.info("Initializing classifier...")
    model_config = MODELS[args.chat_model]
    embedding_config = EMBEDDING_MODELS[args.embedding_model]

    classifier = SOCClassifier(
        chat_model_name=model_config["name"],
        embedding_model_name=embedding_config["name"],
        load_in_8bit=model_config.get("load_in_8bit", False),
        load_in_4bit=model_config.get("load_in_4bit", False),
    )

    logger.info("Classifier initialized successfully")

    # Process rows
    results = []
    checkpoint_file = f"{Path(args.output).stem}_checkpoint.csv"

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        if idx < start_idx:
            continue

        question = row[args.question_col]
        answer = row[args.answer_col]
        row_id = row[args.id_col]

        logger.info(f"Processing row {idx+1}/{len(df)}: ID={row_id}, Answer='{answer}'")

        # Process based on mode
        if args.mode == "classify":
            result = process_row_classify(
                classifier, question, answer,
                args.k, args.collection, args.prompt
            )
        else:  # followup mode
            result = process_row_followup(
                classifier, question, answer,
                args.k, args.collection, args.prompt,
                args.max_followups
            )

        # Add to results
        result_row = {
            args.id_col: row_id,
            args.question_col: question,
            args.answer_col: answer,
            **result
        }
        results.append(result_row)

        # Save checkpoint
        if (idx + 1) % args.checkpoint_every == 0:
            results_df = pd.DataFrame(results)
            save_checkpoint(results_df, checkpoint_file)

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    # Summary statistics
    logger.info("="*80)
    logger.info("Summary Statistics")
    logger.info("="*80)
    logger.info(f"Total rows processed: {len(results_df)}")
    logger.info(f"Successful classifications: {(results_df['soc_code'] != 'NONE').sum()}")
    logger.info(f"Errors: {(results_df['soc_code'] == 'ERROR').sum()}")
    logger.info(f"No classification: {(results_df['soc_code'] == 'NONE').sum()}")

    if args.mode == "followup":
        logger.info(f"Average follow-ups: {results_df['num_followups'].mean():.2f}")

    logger.info("="*80)
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
