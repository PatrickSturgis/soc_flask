#!/usr/bin/env python3
"""Verify ChromaDB collections are properly populated."""
import chromadb

CHROMA_PATH = "/data/spack/users/sturgis/chroma"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collections = client.list_collections()

print("ChromaDB Collections:")
print("=" * 60)
for coll in collections:
    print(f"\nCollection: {coll.name}")
    print(f"  Count: {coll.count()}")
    
    # Test query
    if coll.count() > 0:
        results = coll.query(
            query_embeddings=[[0.1] * 768],  # Dummy embedding
            n_results=3
        )
        print(f"  Sample IDs: {results['ids'][0][:3]}")
        if results['documents']:
            print(f"  Sample docs: {results['documents'][0][:3]}")

print("\n" + "=" * 60)
print("✓ Verification complete")
