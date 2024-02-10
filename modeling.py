# Import necessary libraries and modules
import pandas as pd
import os
from tqdm import tqdm
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertForSequenceClassification, AlbertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from optimizer_loss import CrossEntropyLoss, FocalLossWithBatchNormL2
from optimizer_loss import LabelSmoothingLoss, FocalLoss
from logger_config import logger

from enum import Enum


class LossType(Enum):
    CE = 'ce'
    FCL = 'fcl'
    FCLBNL2 = 'fclbnl2'
    LBSMOOTHINGLOSS = 'lbsmoothingloss'

class MedicalTextOptimizeLoss:
    def __init__(self, args, num_labels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = args.model_pretrain
        self.loss_type = args.loss_type
        self.model_dir = args.model_dir
        self.data_preprocess = args.data_preprocess
        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.num_epochs = args.epochs
        self.step_per_epochs = args.step_per_epoch
        self.max_length = args.max_length
        self.reduce_step_size = args.reduce_step_size
        self.gamma = args.reduce_gamma
        self.scheduler = args.scheduler
        self.truncate = args.truncate
        self.padding = args.padding
        self.report_method = args.report_method
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.output_dir = self.create_output_dir(args.output_dir)

        self.load_model()
        self.select_loss_type()
        self.initialize_optimizer_scheduler()

    def create_output_dir(self, out_dir):
        return f'{out_dir}/{self.model_name}-{self.data_preprocess}-{self.loss_type}-{self.scheduler}.pth'

    def load_model(self):
        logger.info("*** MedicalTextOptimizeLoss: MedicalText Optimization loss creating...")
        direction_model = f'{self.model_dir}/{self.model_name}'

        if self.model_name.lower() == 'albert_base_v2':
            self.tokenizer = AlbertTokenizer.from_pretrained(direction_model)
            self.model = AlbertForSequenceClassification.from_pretrained(direction_model, num_labels=self.num_labels)
        elif self.model_name.lower() == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained(direction_model)
            self.model = RobertaForSequenceClassification.from_pretrained(direction_model, num_labels=self.num_labels)
        elif self.model_name.lower() == 'clinicalbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(direction_model)
            self.model = DistilBertForSequenceClassification.from_pretrained(direction_model, num_labels=self.num_labels)
        elif os.path.exists(self.model_dir):
            self.tokenizer = BertTokenizer.from_pretrained(direction_model)
            self.model = BertForSequenceClassification.from_pretrained(direction_model, num_labels=self.num_labels)
        else:
            raise ValueError(f"Unsupported or Path not correct: Name: {self.model_name}, Path: {self.model_dir}")

        self.model.to(self.device)
        logger.info("*** MedicalTextOptimizeLoss: MedicalText Optimization loss was loaded.")

    def select_loss_type(self):
        if self.loss_type == LossType.CE.value:
            self.loss_instance = CrossEntropyLoss()
        elif self.loss_type == LossType.FCL.value:
            self.loss_instance = FocalLoss()
        elif self.loss_type == LossType.FCLBNL2.value:
            self.loss_instance = FocalLossWithBatchNormL2()
        elif self.loss_type == LossType.LBSMOOTHINGLOSS.value:
            self.loss_instance = LabelSmoothingLoss()
        else:
            raise ValueError('Loss functions must be in [ce, fcl, fclbnl2, lbsmoothingloss]')

    def initialize_optimizer_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.scheduler:
            self.scheduler_prc = StepLR(self.optimizer, step_size=self.reduce_step_size, gamma=self.gamma)
        logger.info(f"*** MedicalTextOptimizeLoss: Successfully created initialize_optimizer with scheduler: {self.scheduler}")

class MedicalTextClassifier(MedicalTextOptimizeLoss):
    def __init__(self, args, num_labels):
        super().__init__(args, num_labels)

    def fit_data(self, data_train, data_test):
        train_inputs, train_labels = self.tokenize_and_format(data_train)
        test_inputs, test_labels = self.tokenize_and_format(data_test)

        train_dataset = self.create_dataset(train_inputs, train_labels)
        test_dataset = self.create_dataset(test_inputs, test_labels)

        self.create_data_loaders(train_dataset, test_dataset)

    def tokenize_and_format(self, data):
        inputs = self.tokenizer([item[0] for item in data],
                                return_tensors='pt',
                                padding=self.padding,
                                truncation=self.truncate,
                                max_length=self.max_length)
        labels = torch.tensor([item[1] for item in data], dtype=torch.long).to(self.device)
        return inputs, labels

    def create_dataset(self, inputs, labels):
        if self.model_name.lower() not in ['roberta-base', 'clinicalbert']:
            return TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], labels)
        else:
            return TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

    def create_data_loaders(self, train_dataset, test_dataset):
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        logger.info("*** MedicalTextClassifier: Successfully created train-test DataLoader")

    def train_and_evaluate(self):
        logger.info("*** MedicalTextClassifier: Training and Evaluating...")
        best_statistic = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_batches = len(self.train_dataloader)
            total_correct_train = 0
            total_samples_train = 0

            for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=total_batches, desc=f"Epoch(s) {epoch + 1}"):
                batch = tuple(t.to(self.device) for t in batch)

                if self.model_name.lower() in ['roberta-base', 'clinicalbert']:
                    (input_ids, attention_mask, labels) = batch
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    (input_ids, attention_mask, token_type_ids, labels) = batch
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

                loss = self.loss_instance(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    logits = outputs.logits
                    _, predicted_labels = torch.max(logits, 1)

                total_correct_train += (predicted_labels == labels).sum().item()
                total_samples_train += labels.size(0)

                if batch_idx % self.step_per_epochs == 0:
                    logger.info(f"\t Batch {batch_idx }/{total_batches}, \t loss: {loss.item():.4f}")

            accuracy_train = total_correct_train / total_samples_train
            logger.info(f"\t Accuracy training at Epoch(s) {epoch + 1}: {accuracy_train:.4f}")

            if self.scheduler:
                self.scheduler_prc.step()
                logger.info(f"\t Scheduler learning rate: {self.scheduler_prc.get_last_lr()}")

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_dataloader):
                    batch = tuple(t.to(self.device) for t in batch)

                    if self.model_name.lower() in ['roberta-base', 'clinicalbert']:
                        (input_ids, attention_mask, labels) = batch
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        (input_ids, attention_mask, token_type_ids, labels) = batch
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                    logits = outputs.logits
                    _, predicted_labels = torch.max(logits, 1)

                    all_preds.extend(predicted_labels.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            statistic = (f1_score(all_labels, all_preds, average=self.report_method) * 100).round(2)

            if self.report_method == 'micro':
                if best_statistic < statistic:
                    best_statistic = statistic
                    torch.save(self.model.state_dict(), self.output_dir)
                    logger.info(f"\t Save the Best model at Epoch(s) {epoch + 1} - saved output: {self.output_dir}")

            logger.info(f"\t Epoch(s) {epoch + 1} - Classification statistic: {statistic}")

        logger.info(f'Best statistic for testing with [{self.report_method}] is: {best_statistic}')