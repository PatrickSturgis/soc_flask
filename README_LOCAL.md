# SOC20 Classifier - Local Version for Winston

This is the local version of the SOC20 classification pipeline, designed to run on Winston using open-source models instead of OpenAI API.

## Overview

**Key differences from the API version:**
- Uses **Qwen2.5** models (7B/14B/32B/72B) via HuggingFace transformers
- Uses **sentence-transformers** for embeddings (e.g., all-mpnet-base-v2)
- Uses **ChromaDB** for local vector storage (replaces Pinecone)
- Designed for **batch processing** via Slurm jobs
- **No external API dependencies** - completely local

## Hardware Requirements

Winston specs:
- 2x RTX 6000 Ada GPUs (48GB VRAM each)
- 256GB RAM
- 96 CPU cores

Recommended model configurations:
- **Qwen2.5-7B**: 1 GPU, 32GB RAM, ~16GB VRAM
- **Qwen2.5-14B**: 1 GPU, 48GB RAM, ~30GB VRAM
- **Qwen2.5-32B-8bit**: 1 GPU, 64GB RAM, ~35GB VRAM
- **Qwen2.5-72B-8bit**: 2 GPUs, 128GB RAM, ~80GB VRAM

## Setup Instructions

### 1. Transfer files to Winston

```bash
# On your local machine
scp -r /path/to/soc_flask hod123@winston:/data/spack/users/hod123/
```

### 2. Create virtual environment on Winston

```bash
# SSH to Winston
ssh hod123@winston

# Navigate to project directory
cd /data/spack/users/hod123/soc_flask

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements_local.txt
```

### 3. Initialize ChromaDB

You need to populate ChromaDB with SOC codes and job title embeddings:

```bash
# Option 1: Using existing embeddings CSV files
python init_chromadb.py \
    --chroma-path /data/spack/users/hod123/chroma \
    --soc-file soc_2020.csv \
    --embeddings-file openai_soc_embeds_l3072.csv \
    --collection-name job_titles_4d \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --reset

# Option 2: Just initialize SOC codes (for testing)
python init_chromadb.py \
    --chroma-path /data/spack/users/hod123/chroma \
    --soc-file soc_2020.csv \
    --embedding-model sentence-transformers/all-mpnet-base-v2
```

**Note:** The existing CSV files (`openai_soc_embeds_l3072.csv`, `pc_embeds_oai_l3072.csv`) contain OpenAI embeddings with 3072 dimensions. You may need to regenerate embeddings using sentence-transformers models which typically have 384-1024 dimensions.

### 4. Test the classifier

```bash
# Quick test with Python
python soc_classifier_local.py
```

This will run a simple test classification.

## Usage

### Batch Processing

Process a CSV file of survey responses:

```bash
python batch_classify.py \
    --input survey_responses.csv \
    --output results.csv \
    --chat-model qwen-7b \
    --embedding-model all-mpnet-base-v2 \
    --k 10 \
    --mode classify
```

**Input CSV format:**
```csv
id,question,answer
1,"What was your main job title?","software developer"
2,"What was your main job title?","nurse"
```

**Output CSV includes:**
- All input columns
- `soc_code`: The assigned 4-digit SOC code
- `soc_desc`: Description of the SOC code
- `soc_conf`: Confidence score (0-100)
- `soc_cands`: Candidate codes considered
- `response`: Full model response
- `error`: Any error messages

### Slurm Job Submission

```bash
# Create logs directory
mkdir -p logs

# Submit basic job
sbatch run_soc_classify.sh

# Submit with custom parameters
sbatch --export=ALL,MODEL=qwen-14b,INPUT=data.csv,OUTPUT=results.csv run_soc_classify.sh

# Submit with limit (for testing)
sbatch --export=ALL,MODEL=qwen-7b,LIMIT=100 run_soc_classify.sh
```

### Interactive Session

For testing or development:

```bash
# Request interactive GPU session
srun --partition=gpu --gres=gpu:1 --mem=48G --cpus-per-task=8 --time=02:00:00 --pty bash

# Activate environment
cd /data/spack/users/hod123/soc_flask
source venv/bin/activate

# Run classification
python batch_classify.py --input test.csv --output results.csv --chat-model qwen-7b --limit 10
```

## Configuration

### Model Selection

Edit `config.py` to modify model configurations or add new models:

```python
MODELS = {
    "qwen-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "load_in_8bit": False,
        "load_in_4bit": False,
    },
    # Add more models...
}
```

### Prompts

Prompts are stored in `static/` directory:
- `force_classify.txt`: Forces classification without follow-ups
- `followup_prompt.txt`: Allows follow-up questions

## Modes

### 1. Classify Mode (Default)

Forces a classification for every input:

```bash
python batch_classify.py \
    --input data.csv \
    --output results.csv \
    --mode classify \
    --prompt force_classify.txt
```

### 2. Follow-up Mode

Allows the model to ask follow-up questions (limited in batch mode):

```bash
python batch_classify.py \
    --input data.csv \
    --output results.csv \
    --mode followup \
    --prompt followup_prompt.txt \
    --max-followups 3
```

**Note:** In batch mode, follow-up questions cannot be answered automatically. For interactive classification, you'll need to implement a separate script.

## Performance Tips

### Model Selection
- **Testing/development**: Use `qwen-7b` or `qwen-7b-8bit`
- **Production**: Use `qwen-14b` or `qwen-32b-8bit` for better accuracy
- **Best accuracy**: Use `qwen-72b-8bit` (requires 2 GPUs)

### Batch Size
Currently processes one row at a time. For future optimization, consider:
- Batching multiple inputs together
- Using vLLM for faster inference
- Pre-computing embeddings for common job titles

### Checkpointing
The script automatically saves checkpoints every 100 rows. To resume:

```bash
python batch_classify.py \
    --input data.csv \
    --output results.csv \
    --resume-from results_checkpoint.csv
```

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA OOM errors:

1. Use quantized models (8-bit or 4-bit):
```bash
--chat-model qwen-7b-8bit
```

2. Request more memory in Slurm:
```bash
sbatch --mem=64G run_soc_classify.sh
```

3. Use a smaller model:
```bash
--chat-model qwen-7b
```

### Slow Inference

1. Check GPU utilization:
```bash
nvidia-smi
```

2. Ensure model is on GPU:
- Check logs for "Using device: cuda"

3. Consider using vLLM for production (requires additional setup)

### ChromaDB Issues

If ChromaDB queries fail:

1. Verify ChromaDB is initialized:
```bash
python init_chromadb.py --chroma-path /data/spack/users/hod123/chroma
```

2. Check collection exists:
```python
import chromadb
client = chromadb.PersistentClient(path="/data/spack/users/hod123/chroma")
print(client.list_collections())
```

3. Verify embedding dimensions match:
- sentence-transformers models: 384-1024 dims
- OpenAI embeddings: 1536 or 3072 dims
- Collections must use consistent dimensions

## Comparison with API Version

| Feature | API Version (app.py) | Local Version |
|---------|---------------------|---------------|
| Chat Model | OpenAI GPT-4/GPT-3.5 | Qwen2.5 (7B-72B) |
| Embeddings | OpenAI text-embedding-3-large | sentence-transformers |
| Vector DB | Pinecone | ChromaDB |
| Interface | Flask REST API | Batch script / Python class |
| Deployment | Render (cloud) | Winston (Slurm) |
| Cost | Pay per API call | Free (local compute) |
| Speed | Fast (optimized API) | Depends on model size |
| Privacy | Data sent to OpenAI | Fully local |

## Next Steps

1. **Optimize embedding generation**: Pre-compute embeddings for all job titles in your dataset
2. **Implement vLLM**: For faster inference in production
3. **Create interactive mode**: Allow manual follow-up questions
4. **Evaluation**: Compare accuracy with OpenAI version
5. **Fine-tuning**: Consider fine-tuning Qwen2.5 on SOC classification examples

## Files Overview

```
soc_flask/
├── app.py                      # Original Flask API (OpenAI + Pinecone)
├── soc_classifier_local.py     # Local classifier class
├── batch_classify.py           # Batch processing script
├── init_chromadb.py           # ChromaDB initialization
├── config.py                   # Model configurations
├── run_soc_classify.sh        # Slurm job script
├── requirements_local.txt      # Local dependencies
├── static/
│   ├── followup_prompt.txt    # Follow-up mode prompt
│   └── force_classify.txt     # Classify mode prompt
└── logs/                       # Job logs (created automatically)
```

## Support

For questions or issues:
1. Check the logs in `logs/` directory
2. Review Winston's Slurm documentation
3. Consult the HuggingFace transformers documentation

## License

Same as the original SOC classification pipeline.
