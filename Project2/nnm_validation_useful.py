import pickle
import torch
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer

# Load the trained model from the pickle file
with open('neural_network_model_project2_useful.pkl', 'rb') as f:
    model = pickle.load(f)

# Load validation data from JSON file
data = pd.read_json("./yelp_reviews_validation_set.json")
data = data.dropna(subset=['useful'])
data = data[data['useful'] != '']
data['useful'] = pd.to_numeric(data['useful'])  # Convert 'cool' column to numeric if it's not already

# Tokenize and encode the text data
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
max_len = 120
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, max_length=max_len, truncation=True) for text in
                   data['text']]
padded_sequences = torch.tensor([text + [0] * (max_len - len(text)) for text in tokenized_texts])

# Convert labels to tensor
labels = torch.tensor(data['useful'])  # Subtract 1 to convert labels to start from 0 for classification

# Create DataLoader for validation
val_data = TensorDataset(padded_sequences, labels)
val_sampler = SequentialSampler(val_data)
batch_size = 32
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Validate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

true_labels = []
predicted_labels = []

with torch.no_grad():
    for batch in tqdm(val_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        outputs = model(inputs)
        _, predicted = torch.max(outputs.logits, 1)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate metrics
true_labels = [label.item() for label in true_labels]
accuracy = accuracy_score(true_labels, predicted_labels) * 100
micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

# Print metrics
print("Accuracy: {:.2f}%".format(accuracy))
print("Micro F1 Score: {:.4f}".format(micro_f1))
print("Macro F1 Score: {:.4f}".format(macro_f1))
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))
