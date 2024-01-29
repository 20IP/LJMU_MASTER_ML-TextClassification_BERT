#!/bin/bash

# List of model names
model_names=("bert-base-uncased" "roberta-base" "bluebert_pubmed_uncased" "bioBERT_v1.1" "clinicalBERT" "albert_base_v2")

# Loop over each model name and run the Python script
for model_name in "${model_names[@]}"; do
    echo "Running script with model_name: $model_name"
    python training.py --model_pretrain "$model_name"
    echo "Finished running script with model_name: $model_name"
done

# chmod +x run_script.sh
# ./run_script.sh