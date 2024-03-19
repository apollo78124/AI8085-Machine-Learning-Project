import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Step 2: Define Dataset and DataLoader
class CustomDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=128):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        score = self.scores[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'score': torch.tensor(score, dtype=torch.float)
        }



# Step 4: Define Neural Network Model
class ScorePredictionModel(nn.Module):
    def __init__(self, bert_model):
        super(ScorePredictionModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        score = self.linear(pooled_output)
        return score

# Step 1: Load and Preprocess Data
# Assuming 'data.json' contains the JSON data with "score" and "text" fields
with open('./yelp_reviews_purified_3steps.json', 'r', encoding='utf-8', errors='ignore') as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")

texts = [item['text'] for item in data]
scores = [item['stars'] for item in data]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = CustomDataset(texts, scores, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=None)

# Step 3: Load Pre-trained BERT Model
bert_model = BertModel.from_pretrained('bert-base-uncased')

model = ScorePredictionModel(bert_model)

# Step 5: Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.MSELoss()
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_scores = batch['score'].to(device)

        optimizer.zero_grad()
        predicted_scores = model(input_ids, attention_mask).squeeze()
        loss = criterion(predicted_scores, target_scores[0])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Save trained model
torch.save(model.state_dict(), 'star_prediction_model.pth')