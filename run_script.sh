#!/bin/bash
# List of model names
model_names=("bert-base-uncased" "roberta-base" "bioBERT_v1.1" "alBERT_base_v2" "bluebert_pubmed_uncased" "clinicalBERT")

# List of loss functions
loss_function_list=("ce" "fcl" "fclbnl2" "lbsmoothingloss")

# Loop over each loss function
for loss_function in "${loss_function_list[@]}"; do
    echo "Running scripts with loss function: $loss_function"

    # Loop over each model name
    for model_name in "${model_names[@]}"; do
        echo "Running script with model_pretrain: $model_name, loss function: $loss_function"

        # Clear cache memory
        sudo sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'
        sudo sh -c 'sync; echo 2 > /proc/sys/vm/drop_caches'

        # Kill existing Python processes
        pkill -f "python training.py --model_pretrain $model_name"
        
        # Run the Python script with different combinations of parameters
        python training.py --model_pretrain "$model_name" --data_preprocess=$false --learning_rate 5e-5 --scheduler=$false --loss_type "$loss_function" --max_length 512

        echo "Finished running script with model_pretrain: $model_name, loss function: $loss_function"
        
    done
done
