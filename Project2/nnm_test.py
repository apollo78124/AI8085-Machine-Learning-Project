import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load saved model
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
model.load_state_dict(torch.load("bert_sentiment_model.pth"))
model.eval()

# Input string for prediction
input_string = "REALLY below average food.  Akin to waffle house.  I wanted this to be good based on the reviews but not so.  The pancakes tasted like a frozen processed version, \"homemade\" maple syrup was watery/sugary with fake flavoring.  Trees make authentic syrup, not people.  The hash browns were greasy and fairly tasteless. The bacon overcooked, hard and chewy.  Obviously had been sitting awhile. Overall, very uninspired food.  Unfortunately, this is one of those places that is living on an outdated \"must go\" reputation.  There are much better offerings in Nashville."

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