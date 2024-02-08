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

from optimizer_loss import CrossEntropyLoss, FocalLoss, FocalLossWithBatchNormL2
from optimizer_loss import LabelSmoothingLoss

from enum import Enum
from logger_config import logger

class LossType(Enum):
    CE = 'ce'
    FCL = 'fcl'
    FCLBNL2 = 'fclbnl2'
    LBSMOOTHINGLOSS = 'lbsmoothingloss'

# Define a class for handling Medical Text Optimization and Loss
class MedicalTextOptimizeLoss:
    def __init__(self, args, num_labels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.model_name = args.model_pretrain
        self.loss_type = args.loss_type
        self.model_dir = args.model_dir
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
        self.data_preprocess = args.data_preprocess
        self.report_method = args.report_method
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.output_dir = self.create_output_dir()

        self.load_model()
        self.select_loss_type()
        self.initialize_optimizer_scheduler()

    def create_output_dir(self):
        return f'{self.output_dir}/{self.model_name}-{self.data_preprocess}-{self.loss_type}-{self.scheduler}.pth'
    
    def load_model(self):
        """
        Load the pre-trained model and tokenizer based on the specified model name.
        """
        logger.info("*** MedicalTextOptimizeLoss: MedicalText Optimization loss creating...")
        direction_model = f'{self.model_dir}/{self.model_name}'
        
        # Check for the specified model name and load the corresponding model and tokenizer
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
        
        # Move the model to the specified device (GPU or CPU)
        self.model.to(self.device)
        
        logger.info("*** MedicalTextOptimizeLoss: MedicalText Optimization loss was loaded.")
                    
    def select_loss_type(self):
        """
        Select the loss function based on the specified loss type.
        """
        # Choose the appropriate loss function based on the specified loss type
        if self.loss_type == 'ce':
            self.loss_instance = CrossEntropyLoss()
        elif self.loss_type == 'fcl':
            self.loss_instance = FocalLoss()
        elif self.loss_type == 'fclbnl2':
            self.loss_instance = FocalLossWithBatchNormL2()
        elif self.loss_type == 'lbsmoothingloss':
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
        """
        Initialize the MedicalTextClassifier class, inheriting from MedicalTextOptimizeLoss.
        """
        super().__init__(args, num_labels)

    def fit_data(self, data_train, data_test):
        """
        Prepare and load training and testing data for the classifier.
        """
        # Tokenize and format the input data for training
        train_inputs = self.tokenizer([item[0] for item in data_train],
                                      return_tensors='pt',
                                      padding=self.padding,
                                      truncation=self.truncate,
                                      max_length=self.max_length)
        train_labels = torch.tensor([item[1] for item in data_train], dtype=torch.long).to(self.device)

        # Tokenize and format the input data for testing
        test_inputs = self.tokenizer([item[0] for item in data_test],
                                     return_tensors='pt',
                                     padding=self.padding,
                                     truncation=self.truncate,
                                     max_length=self.max_length)
        test_labels = torch.tensor([item[1] for item in data_test], dtype=torch.long)

        # Create TensorDatasets based on the model type
        if self.model_name.lower() not in ['roberta-base', 'clinicalbert']:
            train_dataset = TensorDataset(train_inputs['input_ids'],
                                          train_inputs['attention_mask'],
                                          train_inputs['token_type_ids'],
                                          train_labels)
            test_dataset = TensorDataset(test_inputs['input_ids'],
                                         test_inputs['attention_mask'],
                                         test_inputs['token_type_ids'],
                                         test_labels)
        else:
            train_dataset = TensorDataset(train_inputs['input_ids'],
                                          train_inputs['attention_mask'],
                                          train_labels)
            test_dataset = TensorDataset(test_inputs['input_ids'],
                                         test_inputs['attention_mask'],
                                         test_labels)

        # Create DataLoader for training and testing
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False)
        logger.info("*** MedicalTextClassifier: Successfully created train-test DataLoader")

    def train_and_evaluate(self):
        """
        Train and evaluate the classifier using the loaded data.
        """
        logger.info("*** MedicalTextClassifier: Training and Evaluating...")
        best_statistic = 0

        # Iterate through epochs
        for epoch in range(self.num_epochs):
            self.model.train()
            total_batches = len(self.train_dataloader)
            total_correct_train = 0
            total_samples_train = 0

            # Iterate through batches in training data
            for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=total_batches, desc=f"Epoch(s) {epoch + 1}"):
                batch = tuple(t.to(self.device) for t in batch)
                
                # Extract inputs based on model type
                if self.model_name.lower() in ['roberta-base', 'clinicalbert']:
                    (input_ids, attention_mask, labels) = batch
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
                else:
                    (input_ids, attention_mask, token_type_ids, labels) = batch
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         labels=labels)

                # Calculate loss, backpropagate, and update parameters
                loss = self.loss_instance(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()

                # Evaluate training accuracy
                with torch.no_grad():
                    logits = outputs.logits
                    _, predicted_labels = torch.max(logits, 1)

                total_correct_train += (predicted_labels == labels).sum().item()
                total_samples_train += labels.size(0)

            # Log batch loss
            if batch_idx % self.step_per_epochs == 0:
                logger.info(f"\t Batch {batch_idx + 1}/{total_batches}, \t loss: {loss.item():.4f}")

            # Calculate and log training accuracy
            accuracy_train = total_correct_train / total_samples_train
            logger.info(f"\t Accuracy training at Epoch(s) {epoch + 1}: {accuracy_train:.4f}")
            
            # Update learning rate with scheduler
            if self.scheduler:
                self.scheduler_prc.step()
                logger.info(f"\t Scheduler learning rate: {self.scheduler_prc.get_last_lr()}")

            # Evaluate on testing data
            # self.model.eval()
            all_preds = []
            all_labels = []

            # Iterate through batches in testing data
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_dataloader):
                    batch = tuple(t.to(self.device) for t in batch)
                    if self.model_name.lower() in ['roberta-base', 'clinicalbert']:
                        (input_ids, attention_mask, labels) = batch
                        outputs = self.model(input_ids=input_ids,
                                             attention_mask=attention_mask)
                    else:
                        (input_ids, attention_mask, token_type_ids, labels) = batch
                        outputs = self.model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids)

                    # Predict and collect labels
                    logits = outputs.logits
                    _, predicted_labels = torch.max(logits, 1)

                    all_preds.extend(predicted_labels.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Calculate F1 score based on the specified method
            statistic = (f1_score(all_labels, all_preds, average=self.report_method) * 100).round(2)

            # Save the best model based on F1 score (for micro average)
            if self.report_method == 'micro':
                if best_statistic < statistic:
                    best_statistic = statistic
                    torch.save(self.model.state_dict(), self.output_dir)
                    logger.info(f"\t Save the Best model at Epoch(s) {epoch + 1} - saved output: {self.output_dir}")
            # Log classification statistic
            logger.info(f"\t Epoch(s) {epoch + 1} - Classification statistic: {statistic}")

        # Log the best testing accuracy
        logger.info(f'Best statistic for testing with [{self.report_method}] is: {best_statistic}')
