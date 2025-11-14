from soc_classifier_local import SOCClassifier
c = SOCClassifier(chat_model_name='Qwen/Qwen2.5-7B-Instruct', embedding_model_name='sentence-transformers/all-mpnet-base-v2', chroma_path='/data/spack/users/sturgis/chroma')
print('Loaded')
