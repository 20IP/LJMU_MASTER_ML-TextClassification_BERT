#!/bin/bash

# List of model names
# model_names=("bert-base-uncased" "roberta-base" "bioBERT_v1.1" "alBERT_base_v2" "bluebert_pubmed_uncased" "clinicalBERT")
model_names=("clinicalBERT")

# Loop over each model name and run the Python script
for model_name in "${model_names[@]}"; do
    echo "Running script with model_pretrain: $model_name"

    # Clear cache memory
    sudo sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'
    sudo sh -c 'sync; echo 2 > /proc/sys/vm/drop_caches'

    # Kill existing Python processes
    pkill -f "python training.py --model_pretrain $model_name"

    # Run the Python script
    python training.py --model_pretrain "$model_name"

    echo "Finished running script with model_pretrain: $model_name"
done
