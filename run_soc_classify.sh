#!/bin/bash
#SBATCH --job-name=soc_classify
#SBATCH --output=logs/soc_classify_%j.out
#SBATCH --error=logs/soc_classify_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=winston
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# SOC20 Classification Slurm Job Script
# Usage: sbatch run_soc_classify.sh
# Or: sbatch --export=ALL,MODEL=qwen-14b,INPUT=data.csv run_soc_classify.sh

# Configuration (can be overridden via --export when submitting)
MODEL=${MODEL:-qwen-7b-8bit}
INPUT=${INPUT:-r7_parent_vars_cleaned.csv}
OUTPUT=${OUTPUT:-soc_results_$(date +%Y%m%d_%H%M%S).csv}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-all-mpnet-base-v2}
K=${K:-10}
MODE=${MODE:-classify}
LIMIT=${LIMIT:-}
QUESTION_COL=${QUESTION_COL:-fthjobtr7}
ANSWER_COL=${ANSWER_COL:-fthjobtr7}
COLLECTION=${COLLECTION:-job_titles_4d}

# Paths
PROJECT_DIR="/data/spack/users/sturgis/winston_transfer"
CHROMA_PATH="/data/spack/users/sturgis/chroma"
LOG_DIR="${PROJECT_DIR}/logs"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Activate conda environment
source ~/miniconda3/bin/activate soc_env

# Set CUDA visible devices (if needed)
# export CUDA_VISIBLE_DEVICES=0

# Print job information
echo "=========================================="
echo "SOC20 Classification Job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Start time: $(date)"
echo "=========================================="
echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Embedding Model: ${EMBEDDING_MODEL}"
echo "  Input: ${INPUT}"
echo "  Output: ${OUTPUT}"
echo "  Mode: ${MODE}"
echo "  k: ${K}"
if [ -n "${LIMIT}" ]; then
    echo "  Limit: ${LIMIT}"
fi
echo "=========================================="

# Change to project directory
cd "${PROJECT_DIR}" || exit 1

# Run the classification script
python batch_classify.py \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --chat-model "${MODEL}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --question-col "${QUESTION_COL}" \
    --answer-col "${ANSWER_COL}" \
    --collection "${COLLECTION}" \
    --k "${K}" \
    --mode "${MODE}" \
    --checkpoint-every 100 \
    ${LIMIT:+--limit ${LIMIT}}

# Check exit status
EXIT_CODE=$?

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

# Print summary if output file exists
if [ -f "${OUTPUT}" ]; then
    echo "Output file: ${OUTPUT}"
    echo "Number of rows: $(wc -l < ${OUTPUT})"
    echo "First few rows:"
    head -n 5 "${OUTPUT}"
fi

exit ${EXIT_CODE}
