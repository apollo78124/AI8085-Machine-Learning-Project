import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load saved model
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
model.load_state_dict(torch.load("bert_sentiment_model.pth"))
model.eval()

# Input string for prediction
input_string = "I haven't been to the pub for years and was pleasantly surprised. Great atmosphere. Great food. Our server (Kassandra) was terrific. Resolved any issues with elegance and made our dinner great (xtra vinegar great touch). Definitely returning"

# Tokenize input string
tokens = tokenizer.encode_plus(input_string, add_special_tokens=True, max_length=128, truncation=True, return_tensors='pt')

# Perform prediction
with torch.no_grad():
    outputs = model(**tokens)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# Convert predicted class index back to stars value
predicted_stars = predicted_class + 1

print("Predicted stars value:", predicted_stars)