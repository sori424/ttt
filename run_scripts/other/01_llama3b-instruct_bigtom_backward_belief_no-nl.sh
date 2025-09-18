#!/usr/bin/env bash
# Activate conda environment

cd /nethome/julee/miniconda3/bin
source activate gpt-oss
cd /nethome/julee/2025-ttt/ttt

python inference_all.py --model_name "/scratch/common_models/Llama-3.2-3B-Instruct" --task_name "bigtom" --use-rules --rules "data/bigtom/results-backward_belief-false_belief-0.json" 
