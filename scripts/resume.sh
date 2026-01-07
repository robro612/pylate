#!/bin/bash
# Resume compression experiment from results file
# Usage: ./scripts/resume.sh [--save_retrieval_results] [--kmeans_gpu]

python -u scripts/resume_experiment.py \
    --results_file results/compression_experiments_new2/jinaai_jina-colbert-v2/amazon_dataset/beir_format/experiment_20251221_194453/results_20251221_194501.jsonl \
    --save_retrieval_results \
    "$@"