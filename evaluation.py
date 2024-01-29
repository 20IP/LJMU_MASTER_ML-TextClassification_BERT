from transformers import BertForSequenceClassification, BertTokenizer

import torch
import argparse

parser = argparse.ArgumentParser(description="Evaluation Model")
parser.add_argument("--model_dir",
                    type=str,
                    help="Load model to predicting",
                    default='../output-fine-tuning/bert-base-uncased_cross-entropy.pth',
                    required=False)

parser.add_argument("--data_path_test",
                    type=str,
                    default='../Medical-Abstracts-TC-Corpus/medical_tc_test.csv')

class FineTunedBertPredictor:
    def __init__(self, model_path, num_labels):
        # Instantiate the same model architecture used during fine-tuning
        self.model = BertForSequenceClassification(num_labels)

        # Load the fine-tuned model state dict
        self.model.load_state_dict(torch.load(model_path))

        # Put the model in evaluation mode
        self.model.eval()

        # Instantiate the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_data(self, data):
        # Tokenize the input data
        inputs = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=128)
        return inputs

    def predict(self, data):
        # Tokenize the new data
        inputs = self.tokenize_data(data)

        # Make predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).tolist()

        return predictions
# Example usage:
args = parser.parse_args()

model_path = args.model_dir
num_labels = 2  # Replace with the correct number of labels

# Create a Predictor instance
predictor = Predictor(model_path, num_labels)

# New data for prediction
new_data = ["Your new text here.", "Another text to predict."]

# Make predictions
predictions = predictor.predict(new_data)

# Display the predictions
for i, prediction in enumerate(predictions):
    print(f"Prediction for '{new_data[i]}': {prediction}")
