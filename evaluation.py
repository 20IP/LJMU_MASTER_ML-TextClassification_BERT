import argparse
import warnings
import os
from modeling import LossType, CustomDataset

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertForSequenceClassification, AlbertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from optimizer_loss import CrossEntropyLossMultiLabel, FocalLossMultiLabel
from optimizer_loss import FocalLossWithBatchNormL2MultiLabel, LabelSmoothingLossMultiLabel

from dataset import MedicalTextDataEvalLoader

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from utils import calculate_metrics
from enum import Enum


from logger_config import configure_logger
warnings.filterwarnings('ignore')

logger_eval = configure_logger('Evaluation logger.log')

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
                    default='../output-fine-tuning',
                    required=False
                    )

parser.add_argument("--model_name",
                    type=str,
                    default='roberta-base-True-lbsmoothingloss-False.pth',
                    help="chose the model to fine-tuning",
                    required=False
                    )

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    # args.model_name = model_name+lossFunctions+based_process+lemma+scheduler.pth
    model_name = args.model_name
    model_dir = args.model_dir
    data_path = args.data_path
    data_preprocess, loss_type, _ = model_name.replace('.pth', '').split('-')[::-1][-3:]
    
        
    data_loader = MedicalTextDataEvalLoader(data_path, data_preprocess)
    test_data, num_labels = data_loader.load_data(data_type='train')

    direction_model = f'{model_dir}/{model_name}'

    if model_name.lower() == 'albert_base_v2':
        tokenizer = AlbertTokenizer.from_pretrained(direction_model, do_lower_case=True)
        model = AlbertForSequenceClassification.from_pretrained(direction_model, num_labels=num_labels)
    elif model_name.lower() == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(direction_model, do_lower_case=False)
        model = RobertaForSequenceClassification.from_pretrained(direction_model, num_labels=num_labels)
    elif model_name.lower() == 'clinicalbert':
        tokenizer = DistilBertTokenizer.from_pretrained(direction_model, do_lower_case=True)
        model = DistilBertForSequenceClassification.from_pretrained(direction_model, num_labels=num_labels)
    elif model_name.lower() == 'biobert_v1.1':
        tokenizer = BertTokenizer.from_pretrained(direction_model, do_lower_case=False)
        model = BertForSequenceClassification.from_pretrained(direction_model, num_labels=num_labels)
    elif os.path.exists(model_dir):
        tokenizer = BertTokenizer.from_pretrained(direction_model, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(direction_model, num_labels=num_labels)
    else:
        raise ValueError(f"Unsupported or Path not correct: Name: {model_name}, Path: {model_dir}")

    model.to(device)
    logger_eval.info("*** MedicalTextOptimizeLoss: MedicalText Optimization loss was loaded.")

    if loss_type == LossType.CE.value:
        loss_instance = CrossEntropyLossMultiLabel()
    elif loss_type == LossType.FCL.value:
        loss_instance = FocalLossMultiLabel()
    elif loss_type == LossType.FCLBNL2.value:
        loss_instance = FocalLossWithBatchNormL2MultiLabel()
    elif loss_type == LossType.LBSMOOTHINGLOSS.value:
        loss_instance = LabelSmoothingLossMultiLabel()
    else:
        raise ValueError('Loss functions must be in [ce, fcl, fclbnl2, lbsmoothingloss]')
    

    test_inputs = tokenizer(test_data[0],
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    
    test_labels = torch.tensor(test_data[1], dtype=torch.float)
    test_dataset = CustomDataset(test_inputs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predicted_labels = (torch.sigmoid(outputs.logits) >= 0.5).float().cpu().numpy()
            true_lbl = labels.cpu().numpy()
            metric = calculate_metrics(predicted_labels, true_lbl)
            accuracy_test += metric['accuracy']
            f1_score_test += metric['f1_score']
            
    accuracy_test /= len(test_dataloader)
    f1_score_test /= len(test_dataloader)

evaluate()