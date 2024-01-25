import pandas as pd
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, f1_score


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_df = pd.read_csv('medical_tc_train.csv')
test_df = pd.read_csv('medical_tc_test.csv')

# Correctly shift labels by 1 (assuming they start from 1, not 0)
train_df['condition_label'] = train_df['condition_label'] - 1
test_df['condition_label'] = test_df['condition_label'] - 1

train_data = list(zip(train_df['medical_abstract'].tolist(), train_df['condition_label'].tolist()))
test_data = list(zip(test_df['medical_abstract'].tolist(), test_df['condition_label'].tolist()))

num_labels = len(train_df['condition_label'].unique())

local_directory = 'pre-train/clinicalBERT/'
# config = BertConfig.from_pretrained(local_directory)
# print(config)
tokenizer = BertTokenizer.from_pretrained(local_directory)
model = BertForSequenceClassification.from_pretrained(local_directory, num_labels=num_labels)
model.to(device)

train_inputs = tokenizer([item[0] for item in train_data], return_tensors='pt', padding=True, truncation=True, max_length=512)
train_labels = torch.tensor([item[1] for item in train_data], dtype=torch.long).to(device)
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_inputs['token_type_ids'], train_labels)

# Tokenize and encode the sequences for the test set
test_inputs = tokenizer([item[0] for item in test_data], return_tensors='pt', padding=True, truncation=True, max_length=512)
test_labels = torch.tensor([item[1] for item in test_data], dtype=torch.long)
test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_inputs['token_type_ids'], test_labels)

# Create the data loaders for the train and test sets
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

# Train the model for 3 epochs
num_epochs = 4
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_batches = len(train_dataloader)
    total_correct_train = 0
    total_samples_train = 0

    for batch_idx, batch in enumerate(train_dataloader):
        # Move batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Forward pass
        input_ids, attention_mask, token_type_ids, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Track training accuracy
        with torch.no_grad():
            logits = outputs.logits
            _, predicted_labels = torch.max(logits, 1)

        total_correct_train += (predicted_labels == labels).sum().item()
        total_samples_train += labels.size(0)

        # Print progress information
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{total_batches}, Loss: {loss.item():.4f}')

    # Calculate and print training accuracy
    scheduler.step()
    print(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()}')
    accuracy_train = total_correct_train / total_samples_train
    print(f'Epoch {epoch+1}/{num_epochs} - Training Accuracy: {accuracy_train:.4f}')

    # Evaluate the model on the test set after each epoch
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # Move batch to GPU
            batch = tuple(t.to(device) for t in batch)

            input_ids, attention_mask, token_type_ids, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            _, predicted_labels = torch.max(logits, 1)

            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate and print classification report
    report1 = f1_score(all_labels, all_preds, average='micro')
    report2 = classification_report(all_labels, all_preds)
    print(f'Epoch {epoch+1}/{num_epochs} - Classification Report:')
    print(report1)
    print(report2)
