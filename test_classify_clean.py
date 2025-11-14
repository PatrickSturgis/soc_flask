from soc_classifier_local import SOCClassifier
import sys

print("Initializing classifier...")
classifier = SOCClassifier(
    chat_model_name='Qwen/Qwen2.5-7B-Instruct',
    embedding_model_name='sentence-transformers/all-mpnet-base-v2',
    chroma_path='/data/spack/users/sturgis/chroma',
    load_in_8bit=True
)

print("Testing classification...")
result = classifier.classify(
    init_q="What was your main job title?",
    init_ans="teacher",
    k=5,
    collection_name='soc4d'
)

print("SOC Code:", result['soc_code'])
print("Description:", result['soc_desc'])

if result['soc_code'] == 'ERROR':
      print("Full result:", result)
