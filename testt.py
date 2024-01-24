from transformers import BertTokenizer, BertModel

# Specify the local path to the pre-trained model files
local_model_path = '/home/dev/Xavier/LJMU/pre-train/uncased_L-12_H-768_A-12'

# Load pre-trained tokenizer from the local file system
tokenizer = BertTokenizer.from_pretrained(local_model_path)

# Load pre-trained model from the local file system
model = BertModel.from_pretrained(local_model_path)

# You can now use the tokenizer and model for various natural language processing tasks
