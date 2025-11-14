"""
Configuration file for SOC20 local classifier.

Modify these settings based on your needs and available resources.
"""

# Model configurations
MODELS = {
    # Qwen2.5 models (recommended for SOC classification)
    "qwen-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 7B - Fast, good for testing",
        "vram_gb": 16,
        "load_in_8bit": False,
        "load_in_4bit": False,
    },
    "qwen-7b-8bit": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 7B in 8-bit - Memory efficient",
        "vram_gb": 8,
        "load_in_8bit": True,
        "load_in_4bit": False,
    },
    "qwen-7b-4bit": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 7B in 4-bit - Very memory efficient",
        "vram_gb": 5,
        "load_in_8bit": False,
        "load_in_4bit": True,
    },
    "qwen-14b": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "description": "Qwen 14B - Better accuracy",
        "vram_gb": 30,
        "load_in_8bit": False,
        "load_in_4bit": False,
    },
    "qwen-14b-8bit": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "description": "Qwen 14B in 8-bit - Fits on single GPU",
        "vram_gb": 15,
        "load_in_8bit": True,
        "load_in_4bit": False,
    },
    "qwen-32b-8bit": {
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "description": "Qwen 32B in 8-bit - High accuracy",
        "vram_gb": 35,
        "load_in_8bit": True,
        "load_in_4bit": False,
    },
    "qwen-72b-8bit": {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "description": "Qwen 72B in 8-bit - Best accuracy, needs multi-GPU",
        "vram_gb": 80,  # Will use both GPUs with device_map="auto"
        "load_in_8bit": True,
        "load_in_4bit": False,
    },
}

# Embedding model configurations
EMBEDDING_MODELS = {
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "description": "Good general purpose (768 dim)",
        "embedding_dim": 768,
    },
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Fast and lightweight (384 dim)",
        "embedding_dim": 384,
    },
    "all-mpnet-base-v2-alt": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "description": "Alternative path",
        "embedding_dim": 768,
    },
    "multilingual-e5-large": {
        "name": "intfloat/multilingual-e5-large",
        "description": "Multilingual support (1024 dim)",
        "embedding_dim": 1024,
    },
}

# ChromaDB configuration
CHROMA_CONFIG = {
    "path": "/data/spack/users/hod123/chroma",
    "collections": {
        "job_titles_4d": "Collection of job titles mapped to 4-digit SOC codes",
        "soc4d": "Collection of 4-digit SOC codes and descriptions",
    }
}

# Default processing parameters
DEFAULT_PARAMS = {
    "k": 10,  # Number of candidate SOC codes to retrieve
    "collection_name": "job_titles_4d",
    "max_new_tokens": 512,
    "temperature": 0.01,
    "top_p": 1.0,
}

# Slurm resource recommendations
SLURM_CONFIGS = {
    "qwen-7b": {
        "gpus": 1,
        "mem_gb": 32,
        "time": "02:00:00",
    },
    "qwen-14b": {
        "gpus": 1,
        "mem_gb": 48,
        "time": "04:00:00",
    },
    "qwen-32b-8bit": {
        "gpus": 1,
        "mem_gb": 64,
        "time": "06:00:00",
    },
    "qwen-72b-8bit": {
        "gpus": 2,
        "mem_gb": 128,
        "time": "08:00:00",
    },
}

# Paths
STATIC_DIR = "static"
PROMPTS = {
    "followup": "followup_prompt.txt",
    "force_classify": "force_classify.txt",
}
