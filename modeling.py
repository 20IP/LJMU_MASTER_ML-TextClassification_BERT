# Import necessary libraries and modules
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertForSequenceClassification, AlbertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import StepLR

from optimizer_loss import CrossEntropyLossMultiLabel, FocalLossMultiLabel
from optimizer_loss import LabelSmoothingLossMultiLabel
from logger_config import logger

from utils import calculate_metrics
from enum import Enum

class LossType(Enum):
    CE = 'ce'
    FCL = 'fcl'
    LBSMOOTHINGLOSS = 'lbsmoothingloss'

class MedicalTextOptimizeLoss:
    def __init__(self, args, num_labels):
        """
        Initialize MedicalTextOptimizeLoss class.
        Args:
        - args: Arguments for configuration.
        - num_labels (int): Number of labels for classification.
        """
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = args.model_pretrain
        self.loss_type = args.loss_type
        self.model_dir = args.model_dir
        self.based_process = args.based_process
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
        self.lemma = args.data_lemma
        self.num_labels = num_labels
        self.threshold = args.threshold
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
        """
        Create output directory path based on configuration parameters.
        Args:
        - out_dir (str): Base output directory path.

        Returns:
        - str: Complete output directory path.
        """
        return f'{out_dir}/{self.model_name}-{self.loss_type}-{self.based_process}-{self.scheduler}-{self.lemma}.pth'

    def load_model(self):
        """
        Load pre-trained model based on the specified model name.
        """
        
        logger.info("*** MedicalTextOptimizeLoss: MedicalText Optimization loss creating...")
        direction_model = f'{self.model_dir}/{self.model_name}'

        if self.model_name.lower() == 'albert_base_v2':
            self.tokenizer = AlbertTokenizer.from_pretrained(direction_model, do_lower_case=True)
            self.model = AlbertForSequenceClassification.from_pretrained(direction_model, num_labels=self.num_labels)
        elif self.model_name.lower() == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained(direction_model, do_lower_case=False)
            self.model = RobertaForSequenceClassification.from_pretrained(direction_model, num_labels=self.num_labels)
        elif self.model_name.lower() == 'clinicalbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(direction_model, do_lower_case=True)
            self.model = DistilBertForSequenceClassification.from_pretrained(direction_model, num_labels=self.num_labels)
        elif self.model_name.lower() == 'biobert_v1.1':
            self.tokenizer = BertTokenizer.from_pretrained(direction_model, do_lower_case=False)
            self.model = BertForSequenceClassification.from_pretrained(direction_model, num_labels=self.num_labels)
        elif os.path.exists(self.model_dir):
            self.tokenizer = BertTokenizer.from_pretrained(direction_model, do_lower_case=True)
            self.model = BertForSequenceClassification.from_pretrained(direction_model, num_labels=self.num_labels)
        else:
            raise ValueError(f"Unsupported or Path not correct: Name: {self.model_name}, Path: {self.model_dir}")

        self.model.to(self.device)
        logger.info("*** MedicalTextOptimizeLoss: MedicalText Optimization loss was loaded.")

    def select_loss_type(self):
        """
        Select loss function based on the specified loss type.
        """
        
        if self.loss_type == LossType.CE.value:
            self.loss_instance = CrossEntropyLossMultiLabel()
        elif self.loss_type == LossType.FCL.value:
            self.loss_instance = FocalLossMultiLabel()
        elif self.loss_type == LossType.LBSMOOTHINGLOSS.value:
            self.loss_instance = LabelSmoothingLossMultiLabel()
        else:
            raise ValueError('Loss functions must be in [ce, fcl, lbsmoothingloss]')

    def initialize_optimizer_scheduler(self):
        """
        Initialize optimizer and scheduler based on configuration parameters.
        """
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.scheduler:
            self.scheduler_prc = StepLR(self.optimizer, step_size=self.reduce_step_size, gamma=self.gamma)
        logger.info(f"*** MedicalTextOptimizeLoss: Successfully created initialize_optimizer with scheduler: {self.scheduler}")
        

class CustomDataset(Dataset):
    """
    Initialize CustomDataset class.
    Args:
    - tokenized_texts (dict): Tokenized input texts.
    - labels (torch.Tensor): Labels for classification.
    """
    
    def __init__(self, tokenized_texts, labels):
        self.tokenized_texts = tokenized_texts
        self.labels = labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        - int: Length of the dataset.
        """
        
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
        - idx (int): Index of the item.

        Returns:
        - dict: Dictionary containing input_ids, attention_mask, and labels.
        """
        
        input_ids = self.tokenized_texts['input_ids'][idx].to(self.device)  # Move to GPU
        attention_mask = self.tokenized_texts['attention_mask'][idx].to(self.device)  # Move to GPU
        labels = self.labels[idx].to(self.device)  # Move to GPU

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    
class MedicalTextClassifier(MedicalTextOptimizeLoss):
    def __init__(self, args, num_labels):
        """
        Initialize MedicalTextClassifier class.
        Args:
        - args: Arguments for configuration.
        - num_labels (int): Number of labels for classification.
        """
        
        super().__init__(args, num_labels)

    def fit_data(self, data_train, data_test):
        """
        Prepare and tokenize training and testing data.
        Args:
        - data_train: Training data containing texts and labels.
        - data_test: Testing data containing texts and labels.
        """
        
        train_inputs, train_labels = self.tokenize_and_format(data_train)
        test_inputs, test_labels = self.tokenize_and_format(data_test)

        train_dataset = CustomDataset(train_inputs, train_labels)
        test_dataset = CustomDataset(test_inputs, test_labels)

        self.create_data_loaders(train_dataset, test_dataset)

    def tokenize_and_format(self, data):
        """
        Tokenize and format data for input to the model.
        Args:
        - data: Data containing texts and labels.

        Returns:
        - Tuple: Tuple of inputs and labels.
        """
        
        texts, labels = data
        
        inputs = self.tokenizer(texts,
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                                max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float)
        
        return inputs, labels

    def create_dataset(self, inputs, labels):
        """
        Create a PyTorch dataset based on the model name.
        Args:
        - inputs: Input data.
        - labels: Labels for classification.

        Returns:
        - TensorDataset: PyTorch dataset.
        """
        
        if self.model_name.lower() not in ['roberta-base', 'clinicalbert']:
            return TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], labels)
        else:
            return TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

    def create_data_loaders(self, train_dataset, test_dataset):
        """
        Create train and test data loaders for training and evaluation.
        Args:
        - train_dataset: Training dataset.
        - test_dataset: Testing dataset.
        """
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        logger.info("*** MedicalTextClassifier: Successfully created train-test DataLoader")

    def train_and_evaluate(self):
        """
        Train and evaluate the model using the specified loss function and optimizer.
        """
        
        logger.info("*** MedicalTextClassifier: Training and Evaluating...")
        best_f1_score_test = 0
        total_batches_train = len(self.train_dataloader)
        total_batches_test = len(self.test_dataloader)
        for epoch in range(self.num_epochs):
            self.model.train()
            accuracy_train = f1_score_train_micro = f1_score_train_macro = 0
            accuracy_test = f1_score_test_micro = f1_score_test_macro = 0

            for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=total_batches_train, desc=f"Epoch(s) {epoch + 1}"):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
    
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                loss = self.loss_instance(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()

                predicted_labels = (torch.sigmoid(outputs.logits) >= self.threshold).float().cpu().numpy()
                true_lbl = labels.cpu().numpy()
                metric = calculate_metrics(predicted_labels, true_lbl)
                
                accuracy_train += metric['accuracy']
                f1_score_train_micro += metric['f1_score_micro']
                f1_score_train_macro += metric['f1_score_macro']
                
    
                if batch_idx % self.step_per_epochs == 0:
                    logger.info(f"\t Batch {batch_idx }/{total_batches_train}, \t loss: {loss.item():.4f}")
                    
            accuracy_train /= total_batches_train
            f1_score_train_micro /= total_batches_train
            f1_score_train_macro /= total_batches_train
            
            logger.info(f"\t Epoch {epoch + 1} - Training : Accuracy  = {accuracy_train:.4f}, F1-Score Micro = {f1_score_train_micro:.4f}, F1-Score Macro = {f1_score_train_macro:.4f}")
            

            if self.scheduler:
                self.scheduler_prc.step()
                logger.info(f"\t Scheduler learning rate: {self.scheduler_prc.get_last_lr()}")

            # Evaluation with test Data

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_dataloader):
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']

                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                    predicted_labels = (torch.sigmoid(outputs.logits) >= self.threshold).float().cpu().numpy()
                    true_lbl = labels.cpu().numpy()
                    metric = calculate_metrics(predicted_labels, true_lbl)
                    accuracy_test += metric['accuracy']
                    f1_score_test_micro += metric['f1_score_micro']
                    f1_score_test_macro += metric['f1_score_macro']
                    
            accuracy_test /= total_batches_test
            f1_score_test_micro /= total_batches_test
            f1_score_test_macro /= total_batches_test
            
            logger.info(f"\t Epoch {epoch + 1} - Testing : Accuracy Micro  = {accuracy_test:.4f}, F1-Score Micro = {f1_score_test_micro:.4f}, F1-Score Macro = {f1_score_test_macro:.4f}")
            

            if best_f1_score_test < f1_score_test_micro:
                best_f1_score_test = f1_score_test_micro
                torch.save(self.model.state_dict(), self.output_dir)
                logger.info(f"\t Save the best model at epoch(s) {epoch + 1} - saved output: {self.output_dir}")

        logger.info(f"Best statistic for testing: {best_f1_score_test}")
