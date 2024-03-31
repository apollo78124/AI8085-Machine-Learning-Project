import pickle

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# Load data from JSON file
data = pd.read_json("./yelp_reviews_step1.json")
data = data.dropna(subset=['funny'])
data = data[data['funny'] != '']
data['funny'] = pd.to_numeric(data['funny'])  # Convert 'cool' column to numeric if it's not already

# Define BERT model and tokenizer
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Tokenize and encode the text data
max_len = 120  # You can adjust this value according to your needs
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, max_length=max_len, truncation=True) for text in
                   data['text']]

# Pad tokenized sequences
padded_sequences = torch.tensor([text + [0] * (max_len - len(text)) for text in tokenized_texts])

# Convert labels to tensor
labels = torch.tensor(data['funny'] - 1)  # Subtract 1 to convert labels to start from 0 for classification

# Check for invalid labels
num_classes = model.config.num_labels
invalid_indices = labels[labels < 0]  # Check for labels less than 0
if invalid_indices.numel() > 0:
    print("Warning: Found labels less than 0.")
invalid_indices = labels[labels >= num_classes]  # Check for labels greater than or equal to num_classes
if invalid_indices.numel() > 0:
    print("Warning: Found labels greater than or equal to {}.".format(num_classes))

# Remove invalid labels
valid_indices = torch.logical_and(labels >= 0, labels < num_classes)
padded_sequences = padded_sequences[valid_indices]
labels = labels[valid_indices]

# Split data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(padded_sequences, labels, random_state=42,
                                                                      test_size=0.1)

# Create data loaders
batch_size = 32
train_data = TensorDataset(train_inputs, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 1
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc="Epoch {}".format(epoch + 1)):
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to prevent exploding gradients
        optimizer.step()
        scheduler.step()

    # Calculate average training loss
    avg_train_loss = total_train_loss / len(train_dataloader)
    print("Average training loss: {:.2f}".format(avg_train_loss))

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

    # Calculate average validation loss
    avg_val_loss = total_val_loss / len(val_dataloader)
    print("Average validation loss: {:.2f}".format(avg_val_loss))

# Save the trained model
with open('neural_network_model_project2_funny.pkl', 'wb') as f:
    pickle.dump(model, f)
