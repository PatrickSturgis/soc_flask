#!/usr/bin/env python3
"""
Re-embed job titles using sentence-transformers for ChromaDB.
Replaces OpenAI 3072-dim embeddings with sentence-transformers 768-dim embeddings.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# Configuration
CHROMA_PATH = "/data/spack/users/sturgis/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "job_titles_4d"
INPUT_FILE = "openai_soc_embeds_l3072.csv"

print("Loading data...")
# Load the CSV (only need first few columns, not the embeddings)
df = pd.read_csv(INPUT_FILE, usecols=['full_code', 'soc_4', 'title', 'description'])
print(f"Loaded {len(df)} rows")

# Deduplicate to 4-digit level
print("Deduplicating to 4-digit SOC codes...")
df_4d = df.drop_duplicates(subset='soc_4', keep='first')
print(f"After deduplication: {len(df_4d)} unique 4-digit codes")

# Initialize sentence-transformers model
print(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# Initialize ChromaDB
print(f"Connecting to ChromaDB at: {CHROMA_PATH}")
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Delete and recreate collection
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted old collection: {COLLECTION_NAME}")
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Job titles with sentence-transformer embeddings"}
)
print(f"Created collection: {COLLECTION_NAME}")

# Generate embeddings and add to ChromaDB
print("Generating embeddings and loading into ChromaDB...")

ids = []
documents = []
metadatas = []
embeddings_list = []

for idx, row in tqdm(df_4d.iterrows(), total=len(df_4d), desc="Processing"):
    soc_code = str(row['soc_4'])
    title = str(row['title'])

    # Create document text (same format as when querying)
    doc_text = f"{soc_code} - {title}"

    # Generate embedding
    embedding = model.encode(doc_text, show_progress_bar=False)

    ids.append(soc_code)
    documents.append(doc_text)
    metadatas.append({
        "soc_4": soc_code,
        "title": title,
        "desc": doc_text  # For compatibility with existing code
    })
    embeddings_list.append(embedding.tolist())

# Add to ChromaDB in batches
print("Adding to ChromaDB...")
batch_size = 100
for i in tqdm(range(0, len(ids), batch_size), desc="Batches"):
    batch_end = min(i + batch_size, len(ids))
    collection.add(
        ids=ids[i:batch_end],
        documents=documents[i:batch_end],
        metadatas=metadatas[i:batch_end],
        embeddings=embeddings_list[i:batch_end]
    )

print(f"\n✓ Successfully loaded {len(ids)} job titles into '{COLLECTION_NAME}'")
print(f"✓ Collection now has {collection.count()} items")

# Test a query
print("\nTesting query...")
test_query = "teacher"
test_embedding = model.encode(f"Job title: '{test_query}'")
results = collection.query(
    query_embeddings=[test_embedding.tolist()],
    n_results=5
)
print(f"Query: '{test_query}'")
print("Top 5 results:")
for i, (doc_id, doc) in enumerate(zip(results['ids'][0], results['documents'][0])):
    print(f"  {i+1}. {doc}")

print("\n✓ Done!")
