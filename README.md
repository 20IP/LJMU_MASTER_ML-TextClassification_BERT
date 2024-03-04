# Classify Disease Groups Based on Medical Text Context with BERT

![builder](https://github.com/cambridge/thesis/workflows/builder/badge.svg?branch=master&event=push)

>   _Thesis of Master Machine Learning - Liverpool John Moores University_

## Structure of Project
```
├── User Folder
    ├── Medical-Abstracts-TC-Corpus
       └── medical_tc_train.csv
        └── medical_tc_test.csv
        └── medical_tc_labels.csv
        └── README.md
    ├── LJMU_MASTER_ML-TextClassification_BERT
        └── modeling.py
        └── generate_process.py
        └── EDA_data.ipynb
        └── utils.ipynb
        └── Visualizer.ipynb
        └── training.py
        └── dataset.py
        └── optimizer_loss.py
        └── logger_config.py
        └── evaluation.py
        └── requirement.txt
        └── README.md
            ...
    ├── output-fine-tuning
        └── ...(save model during fine-tuning models)
    ├── pre-train
        ├── albert_base_v2
            └── tokenizer_config.json
            └── tokenizer.json
            └── special_tokens_map.json
            └── pytorch_model.bin
                ...
        ├── clinicalBERT
            └── tokenizer_config.json
            └── tokenizer.json
            └── special_tokens_map.json
            └── pytorch_model.bin
                ...
        ├── roberta-base
            └── tokenizer_config.json
            └── tokenizer.json
            └── special_tokens_map.json
            └── pytorch_model.bin
                ...
        ├── bluebert_pubmed_uncased
            └── tokenizer_config.json
            └── tokenizer.json
            └── special_tokens_map.json
            └── pytorch_model.bin
                ...
        ├── bert-base-uncased
            └── tokenizer_config.json
            └── tokenizer.json
            └── special_tokens_map.json
            └── pytorch_model.bin
                ...
        └── bioBERT_v1.1
            └── tokenizer_config.json
            └── tokenizer.json
            └── special_tokens_map.json
            └── pytorch_model.bin
            └── ...
```
# Usage details
## Quick start
* >Ensure file structure as above.

__I. Build the project step by step.__
_Creating an environment with the following steps below:_
1. Clone this repository.

***Using Anaconda:***

2. Create a virtual environment:

```
>> conda create --name your_env_name python=3.8
```
3. Activate the environment:
```
>> conda activate your_env_name
```
4. Install the required libraries:
```
>> pip install -r requirements.txt

```
5. Go to folder `LJMU_MASTER_ML-TextClassification_BERT`
6. Run the file `generate_process.py` to create two new files named `preprocessed-medical_tc_test.csv` and `preprocessed-medical_tc_train.csv`.
Stored in the `Medical-Abstracts-TC-Corpus` folder.

```
>> python generate_process.py --file_name medical_tc_train.csv --output_dir ../Medical-Abstracts-TC-Corpus
```

__II. Introduce the `run_script.sh` file.__
_This file is used to execute multiple models in a specific sequence. The parameters are configured, and the execution is performed as a regular script in the terminal._

**Contents of run_script.sh file** 
```
#!/bin/bash
model_names=("bert-base-uncased" "roberta-base" "bioBERT_v1.1" "alBERT_base_v2" "bluebert_pubmed_uncased" "clinicalBERT")
# List of loss functions
loss_function_list=("ce" "fcl" "lbsmoothingloss")
echo ""
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
        python training.py --model_pretrain "$model_name" --loss_type "$loss_function" --learning_rate 5e-5 --based_process=$false --scheduler=$false --data_lemma=$false --threshold 5e-1 --epochs 3
        echo "Finished running script with model_pretrain: $model_name, loss function: $loss_function"
        echo ""
    done
done
```
**How to configure the parameters (args) in the command: `python training.py --args`**

|ID|based_process|data_lemma|scheduler|Training type|
|---|---|---|---|---|
|0|$false|$false|$false|Using column `medical_abstract` data, `Reduce LR`: false|
|1|$false|$false|true|Using column `medical_abstract` data, `Reduce LR`: true|
|2|true|$false|$false|Using column `normalize_medical_abstract` data, `Reduce LR`: false|
|3|true|$false|true|Using column `normalize_medical_abstract` data, `Reduce LR`: true|
|4|$fasle|true|$false|Using column `lemma_normalize_medical_abstract` data, `Reduce LR `: false|
|5|$fasle|true|true|Using column `lemma_normalize_medical_abstract` data, `Reduce LR`: true|

> * Note:
__This study uses ID = [0, 2, 4] to train the models.__


**The default hyperparameters in model training.`--args`**

|Args|Values|
|:---:|:---:|
|learning_rate|5e-5|
|batch_size|8|
|epochs|3|
|step_per_epoch|200|
|max_length|512|
|threshold|5e-1|
|reduce_step_size|2|
|reduce_gamma|2e-1|

> _Additionally, other hyperparameters such as the number of attention layers and the dropout coefficient, etc, remain consistent with the pre-trained model._


***Fine-Tuned Model Naming Convention***

> _models = {model_name}-{loss_type}-{based_process}-{scheduler}-{lemma}.pth_


7. Set permissions for the file 'run_script.sh'.
```
>> chmod +x run_scripts.sh
```
8. Execute the file 'run_script.sh'.
```
>> ./run_script.sh
```

## Intermediate storage of data, models.

* Store pre-trained models:
> link 1
* Store data.
> link 2
* Store the best model.
> link 3
* Store the repository.
> link 4

## Re-run the model evaluation with the previous model
Assuming you have a previously fine-tuned model such as `alBERT_base_v2-fcl-False-False-False.pth`, to perform model evaluation and obtain statistical metrics, please run the following command:

```
>> python evaluation.py --model_name alBERT_base_v2-fcl-False-False-False.pth
```

**Please ensure that the model truly exists and make sure it is stored in the correct location following the structure above.**


-------------------------------------------------------------------------------

# Troubleshooting
---
# Personal Information

- **Full Name:** PHAM VAN THAI
- **Email:** phamthai.ats@gmail.com
- **Phone:** 0907469308
- **Country:** Vietnam
