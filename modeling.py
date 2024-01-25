import pandas as pd
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import torch

class MedicalTextBase:
    def __init__(self,model_name,
                 lr=5e-5,
                 batch_size=8,
                 num_epochs=4,
                 step_per_epoch=100,
                 max_length = 512,
                 truncate=True,
                 padding = None,
                 scheduler = None,
                 optimize_loss = None
                 ):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_name = model_name
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.step_per_epochs = step_per_epoch
        self.max_length = max_length
        self.scheduler = scheduler
        self.truncate = truncate
        self.padding = padding
        self.optimize_loss = optimize_loss
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.train_dataloader = None
        self.test_dataloader = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        

    def initialize_optimizer_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.5)

    def fit_data(self, data_train, data_test):
        train_inputs = self.tokenizer([item[0] for item in data_train],
                                      return_tensors='pt',
                                      padding=self.padding,
                                      truncation=self.truncate,
                                      max_length=self.max_length)
        
        train_labels = torch.tensor([item[1] for item in data_train], dtype=torch.long).to(self.device)
        train_dataset = TensorDataset(train_inputs['input_ids'],
                                      train_inputs['attention_mask'],
                                      train_inputs['token_type_ids'],
                                      train_labels)

        # Tokenize and encode the sequences for the test set
        test_inputs = self.tokenizer([item[0] for item in data_test],
                                     return_tensors='pt',
                                     padding=self.padding,
                                     truncation=self.truncate,
                                     max_length=self.max_length)
        
        test_labels = torch.tensor([item[1] for item in data_test], dtype=torch.long)
        test_dataset = TensorDataset(test_inputs['input_ids'],
                                     test_inputs['attention_mask'],
                                     test_inputs['token_type_ids'],
                                     test_labels)

        # Create the data loaders for the train and test sets
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True)
        
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False)
        
    def train_and_evaluate(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_batches = len(self.train_dataloader)
            total_correct_train = 0
            total_samples_train = 0

            for batch_idx, batch in enumerate(self.train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)

                # Forward pass
                input_ids, attention_mask, token_type_ids, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                # Track training accuracy
                with torch.no_grad():
                    logits = outputs.logits
                    _, predicted_labels = torch.max(logits, 1)

                total_correct_train += (predicted_labels == labels).sum().item()
                total_samples_train += labels.size(0)

                # Print progress information
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{self.num_epochs}, Batch {batch_idx+1}/{total_batches}, Loss: {loss.item():.4f}')

            # Calculate and print training accuracy
            self.scheduler.step()
            print(f'Epoch {epoch + 1}, Learning Rate: {self.scheduler.get_last_lr()}')
            accuracy_train = total_correct_train / total_samples_train
            print(f'Epoch {epoch+1}/{self.num_epochs} - Training Accuracy: {accuracy_train:.4f}')

            # Evaluate the model on the test set after each epoch
            self.model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_dataloader):
                    # Move batch to GPU
                    batch = tuple(t.to(self.device) for t in batch)

                    input_ids, attention_mask, token_type_ids, labels = batch
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    logits = outputs.logits
                    _, predicted_labels = torch.max(logits, 1)

                    all_preds.extend(predicted_labels.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Calculate and print classification report
            report1 = f1_score(all_labels, all_preds, average='micro')
            report2 = classification_report(all_labels, all_preds)
            print(f'Epoch {epoch+1}/{self.num_epochs} - Classification Report:')
            print(report1)
            print(report2)


class MedicalTextDataLoader:
    def __init__(self, args):
        self.data_path = args.data_path
        self.process_state = args.data_preprocess
        
    def load_data(self):
        train_df = pd.read_csv(f'{self.data_path}/medical_tc_train.csv')
        test_df = pd.read_csv(f'{self.data_path}/medical_tc_test.csv')

        # Correctly shift labels by 1 (assuming they start from 1, not 0)
        train_df['condition_label'] = train_df['condition_label'] - 1
        test_df['condition_label'] = test_df['condition_label'] - 1

        if self.process_state:
            pass
        else:
            pass
        train_data = list(zip(train_df['medical_abstract'].tolist(), train_df['condition_label'].tolist()))
        test_data = list(zip(test_df['medical_abstract'].tolist(), test_df['condition_label'].tolist()))
        print('Data loaded ------------')

        return train_data, test_data