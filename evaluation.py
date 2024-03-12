import argparse
import warnings
import os
from modeling import CustomDataset

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertForSequenceClassification, AlbertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from dataset import MedicalTextDataEvalLoader

import torch
from torch.utils.data import DataLoader
from utils import calculate_metrics


from logger_config import logger
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Initlizing coding")
parser.add_argument("--data_path",
                        type=str,
                        help="data path include train+test file",
                        default='../Medical-Abstracts-TC-Corpus',
                        required=False)

parser.add_argument("--file_name",
                        type=str,
                        help="file name for testing",
                        default='preprocessed-medical_tc_test.csv',
                        required=False)

parser.add_argument("--model_dir",
                    type=str,
                    help="Directory load local output-fine-tuning",
                    default='../pre-train',
                    required=False
                    )

parser.add_argument("--output_dir",
                    type=str,
                    help="Directory load local output-fine-tuning",
                    default='../output-fine-tuning',
                    required=False
                    )

parser.add_argument("--model_finetune",
                    type=str,
                    default='',
                    help="chose the model to fine-tuning",
                    required=True
                    )

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    # args.model_name = {self.model_name}-{self.loss_type}-{self.based_process}-{self.scheduler}-{self.lemma}.pth'
    model_finetune = args.model_finetune
    split_data = model_finetune.replace('.pth', '').split('-')
    _, based_process , _, lemma = split_data[-4], split_data[-3], split_data[-2], split_data[-1]
    model_name = split_data[:-4]
    if len(model_name)>1:
        model_name = '-'.join(model_name)
    else:
        model_name = model_name[0]

    model_dir = args.model_dir
    output_dir = args.output_dir

    args.based_process = eval(based_process)
    args.data_lemma = eval(lemma)

    data_loader = MedicalTextDataEvalLoader(args)
    test_data, num_labels = data_loader.load_data()

    direction_model_pretrain = f'{model_dir}/{model_name}'
    direction_model_finetune = f'{output_dir}/{model_finetune}'

    if model_name.lower() == 'albert_base_v2':
        tokenizer = AlbertTokenizer.from_pretrained(direction_model_pretrain, do_lower_case=True)
        model = AlbertForSequenceClassification.from_pretrained(direction_model_pretrain, num_labels=num_labels)
    elif model_name.lower() == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(direction_model_pretrain, do_lower_case=False)
        model = RobertaForSequenceClassification.from_pretrained(direction_model_pretrain, num_labels=num_labels)
    elif model_name.lower() == 'clinicalbert':
        tokenizer = DistilBertTokenizer.from_pretrained(direction_model_pretrain, do_lower_case=True)
        model = DistilBertForSequenceClassification.from_pretrained(direction_model_pretrain, num_labels=num_labels)
    elif model_name.lower() == 'biobert_v1.1':
        tokenizer = BertTokenizer.from_pretrained(direction_model_pretrain, do_lower_case=False)
        model = BertForSequenceClassification.from_pretrained(direction_model_pretrain, num_labels=num_labels)
    elif os.path.exists(model_dir):
        tokenizer = BertTokenizer.from_pretrained(direction_model_pretrain, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(direction_model_pretrain, num_labels=num_labels)
    else:
        raise ValueError(f"Unsupported or Path not correct: Name: {model_name}, Path: {model_dir}")

    model.load_state_dict(torch.load(direction_model_finetune))
    model.to(device)
    logger.info("*** MedicalTextOptimizeLoss: MedicalText Optimization loss was loaded.")

    test_inputs = tokenizer(test_data[0],
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    
    test_labels = torch.tensor(test_data[1], dtype=torch.float)
    test_dataset = CustomDataset(test_inputs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    accuracy = f1_score_micro = f1_score_macro = 0
    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predicted_labels = (torch.sigmoid(outputs.logits) >= 0.5).float().cpu().numpy()
            true_lbl = labels.cpu().numpy()

            metric = calculate_metrics(predicted_labels, true_lbl)
            accuracy += metric['accuracy']
            f1_score_micro += metric['f1_score_micro']
            f1_score_macro += metric['f1_score_macro']
            
    accuracy /= len(test_dataloader)
    f1_score_micro /= len(test_dataloader)
    f1_score_macro /= len(test_dataloader)
    logger.info(f"*** Accuracy for Evaluation : {accuracy:.4f}")
    logger.info(f"*** Micro F1-scores for Evaluation : {f1_score_micro:.4f}")
    logger.info(f"*** Macro F1-scores for Evaluation : {f1_score_macro:.4f}")

evaluate()
