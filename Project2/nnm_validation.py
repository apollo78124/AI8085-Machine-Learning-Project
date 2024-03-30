import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pandas as pd
from tqdm import tqdm
import pickle

# Load data from JSON file
data = pd.read_json("./yelp_reviews_validation_100k.json")

# Load saved model
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

with open("neural_network_model_project2.pkl", "rb") as f:
    model = pickle.load(f)

# Tokenize and encode the text data
max_len = 512  # You can adjust this value according to your needs
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, max_length=max_len, truncation=True) for text in data['text']]

# Pad tokenized sequences
padded_sequences = torch.tensor([text + [0]*(max_len-len(text)) for text in tokenized_texts])

# Convert labels to tensor
labels = torch.tensor(data['stars'] - 1)  # Subtract 1 to convert star ratings to start from 0

# Create data loader for validation
val_data = TensorDataset(padded_sequences, labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32)

# Validation loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="Validation"):
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        outputs = model(inputs, labels=labels)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = correct / total * 100
print("Total: {:.2f}".format(total))
print("Validation Accuracy: {:.2f}%".format(accuracy))