#!/usr/bin/env python3
"""
Initialize ChromaDB collections for SOC20 classification.

This script helps you populate ChromaDB with:
1. Job titles and their SOC codes (from existing embeddings or fresh generation)
2. SOC code descriptions

Usage:
    python init_chromadb.py --help
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import chromadb

from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize ChromaDB for SOC20 classification"
    )

    parser.add_argument(
        "--chroma-path",
        default="/data/spack/users/hod123/chroma",
        help="Path to ChromaDB persistence directory"
    )

    parser.add_argument(
        "--soc-file",
        default="soc_2020.csv",
        help="CSV file with SOC codes and descriptions"
    )

    parser.add_argument(
        "--embeddings-file",
        help="CSV file with pre-computed embeddings (e.g., openai_soc_embeds_l3072.csv)"
    )

    parser.add_argument(
        "--job-titles-file",
        help="CSV file with job titles to embed (if not using pre-computed embeddings)"
    )

    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model to use for generating embeddings"
    )

    parser.add_argument(
        "--collection-name",
        default="job_titles_4d",
        help="Name for the ChromaDB collection"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and recreate"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for adding embeddings"
    )

    return parser.parse_args()


def load_soc_codes(soc_file: str) -> pd.DataFrame:
    """Load SOC codes and descriptions."""
    logger.info(f"Loading SOC codes from {soc_file}")

    # Try different possible file formats
    if soc_file.endswith('.csv'):
        df = pd.read_csv(soc_file)
    elif soc_file.endswith('.xlsx'):
        df = pd.read_excel(soc_file)
    else:
        raise ValueError(f"Unsupported file format: {soc_file}")

    logger.info(f"Loaded {len(df)} SOC codes")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def create_soc_collection(
    client: chromadb.PersistentClient,
    soc_df: pd.DataFrame,
    collection_name: str,
    embedding_model: SentenceTransformer,
    reset: bool = False
) -> chromadb.Collection:
    """
    Create a ChromaDB collection for SOC codes.

    Args:
        client: ChromaDB client
        soc_df: DataFrame with SOC codes and descriptions
        collection_name: Name for the collection
        embedding_model: Model to generate embeddings
        reset: Whether to delete existing collection

    Returns:
        ChromaDB collection
    """
    # Delete existing collection if reset is True
    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except:
            pass

    # Create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "SOC 2020 codes and descriptions"}
    )

    logger.info(f"Collection '{collection_name}' created/loaded")

    # Determine columns (try common naming patterns)
    code_col = None
    desc_col = None

    for col in soc_df.columns:
        col_lower = col.lower()
        if 'code' in col_lower or 'soc' in col_lower:
            code_col = col
        if 'desc' in col_lower or 'title' in col_lower or 'name' in col_lower:
            desc_col = col

    if not code_col or not desc_col:
        logger.error(f"Could not identify code and description columns")
        logger.error(f"Available columns: {list(soc_df.columns)}")
        raise ValueError("Could not identify required columns")

    logger.info(f"Using columns: code='{code_col}', description='{desc_col}'")

    # Generate embeddings and add to collection
    logger.info("Generating embeddings for SOC descriptions...")

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for idx, row in tqdm(soc_df.iterrows(), total=len(soc_df), desc="Processing SOC codes"):
        code = str(row[code_col])
        desc = str(row[desc_col])

        # Create document text
        doc_text = f"{code} - {desc}"

        # Generate embedding
        embedding = embedding_model.encode(doc_text, show_progress_bar=False)

        ids.append(code)
        documents.append(doc_text)
        metadatas.append({"code": code, "desc": desc})
        embeddings.append(embedding.tolist())

    # Add to collection in batches
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            embeddings=embeddings[i:batch_end]
        )

    logger.info(f"Added {len(ids)} SOC codes to collection")

    return collection


def load_precomputed_embeddings(embeddings_file: str) -> tuple:
    """
    Load pre-computed embeddings from CSV.

    Expected format: First column is ID/code, remaining columns are embedding dimensions.
    """
    logger.info(f"Loading pre-computed embeddings from {embeddings_file}")

    df = pd.read_csv(embeddings_file)

    # Assume first column is ID, rest are embeddings
    ids = df.iloc[:, 0].astype(str).tolist()

    # Extract embedding columns
    embedding_cols = df.columns[1:]
    embeddings = df[embedding_cols].values

    logger.info(f"Loaded {len(ids)} embeddings with dimension {embeddings.shape[1]}")

    return ids, embeddings, df


def main():
    args = parse_args()

    # Create ChromaDB client
    logger.info(f"Initializing ChromaDB at {args.chroma_path}")
    client = chromadb.PersistentClient(path=args.chroma_path)

    # Load embedding model
    logger.info(f"Loading embedding model: {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model)

    # Initialize SOC codes collection
    if args.soc_file:
        soc_df = load_soc_codes(args.soc_file)
        create_soc_collection(
            client,
            soc_df,
            "soc4d",
            embedding_model,
            reset=args.reset
        )

    # Initialize job titles collection
    if args.embeddings_file:
        logger.info("Using pre-computed embeddings")
        ids, embeddings, df = load_precomputed_embeddings(args.embeddings_file)

        # Delete and recreate if reset
        if args.reset:
            try:
                client.delete_collection(args.collection_name)
                logger.info(f"Deleted existing collection: {args.collection_name}")
            except:
                pass

        collection = client.get_or_create_collection(
            name=args.collection_name,
            metadata={"description": "Job titles with pre-computed embeddings"}
        )

        # Add embeddings in batches
        batch_size = args.batch_size
        for i in tqdm(range(0, len(ids), batch_size), desc="Adding to ChromaDB"):
            batch_end = min(i + batch_size, len(ids))

            batch_ids = ids[i:batch_end]
            batch_embeddings = embeddings[i:batch_end].tolist()

            # Create metadata and documents
            batch_metadatas = []
            batch_documents = []

            for j, idx in enumerate(range(i, batch_end)):
                # Extract metadata from dataframe if available
                metadata = {}
                if 'desc' in df.columns:
                    metadata['desc'] = str(df.iloc[idx]['desc'])

                batch_metadatas.append(metadata)
                batch_documents.append(batch_ids[j])

            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )

        logger.info(f"Added {len(ids)} job title embeddings to collection '{args.collection_name}'")

    elif args.job_titles_file:
        logger.info("Generating embeddings from job titles")
        # Implementation for generating fresh embeddings
        # This would be similar to create_soc_collection
        logger.warning("Job titles embedding generation not fully implemented yet")
        logger.warning("Please use --embeddings-file with pre-computed embeddings")

    # Print summary
    logger.info("="*80)
    logger.info("ChromaDB Initialization Complete")
    logger.info("="*80)

    collections = client.list_collections()
    for coll in collections:
        logger.info(f"Collection: {coll.name}")
        logger.info(f"  Count: {coll.count()}")
        logger.info(f"  Metadata: {coll.metadata}")

    logger.info("="*80)


if __name__ == "__main__":
    main()
