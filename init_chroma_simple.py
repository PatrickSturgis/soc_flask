  import chromadb
  import pandas as pd
  from sentence_transformers import SentenceTransformer
  from tqdm import tqdm

  client = chromadb.PersistentClient('/data/spack/users/sturgis/chroma')

  try:
      client.delete_collection('soc4d')
      print("Deleted old collection")
  except:
      pass

  coll = client.create_collection('soc4d')
  df = pd.read_csv('soc_2020_4digit_dedup.csv')

  print(f"Loading embedding model...")
  model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

  print(f"Processing {len(df)} SOC codes...")
  ids = df['soc_4'].astype(str).tolist()
  docs = (df['soc_4'].astype(str) + ' - ' + df['title']).tolist()

  batch_size = 50
  for i in tqdm(range(0, len(ids), batch_size)):
      batch_ids = ids[i:i+batch_size]
      batch_docs = docs[i:i+batch_size]
      batch_embeds = model.encode(batch_docs).tolist()
      coll.add(ids=batch_ids, documents=batch_docs, embeddings=batch_embeds)

  print(f"Done! Added {len(ids)} SOC codes to ChromaDB")

