import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Define the path to the JSON file
json_file_path = r"C:\Users\kunnu\Desktop\COMP 8085\Project 2\yelp_academic_dataset_review.json"

# Initialize empty lists to store data
review_ids = []
user_ids = []
business_ids = []
stars = []
useful = []
funny = []
cool = []
texts = []
dates = []

# Open the JSON file and read the first 20000 records
with open(json_file_path, 'r', encoding='utf-8', errors='ignore') as file:
    for i, line in enumerate(file):
        if i >= 20000:
            break  # Exit the loop after reading 20,000 lines
        try:
            data = json.loads(line)
            review_ids.append(data['review_id'])
            user_ids.append(data['user_id'])
            business_ids.append(data['business_id'])
            stars.append(data['stars'])
            useful.append(data['useful'])
            funny.append(data['funny'])
            cool.append(data['cool'])
            texts.append(data['text'])
            dates.append(data['date'])
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON on line {i + 1}: {e}")

# Create a DataFrame from the extracted data
df = pd.DataFrame({
    'review_id': review_ids,
    'user_id': user_ids,
    'business_id': business_ids,
    'stars': stars,
    'useful': useful,
    'funny': funny,
    'cool': cool,
    'text': texts,
    'date': dates
})

# Data Preprocessing
# Handle missing values
df.dropna(inplace=True)

# Clean text data
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into text
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

df['cleaned_text'] = df['text'].apply(clean_text)

# Additional Data Preprocessing
# Remove non-ASCII characters
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ''.join([i if ord(i) < 128 else '' for i in x]))

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Exploratory Data Analysis (EDA)
# Visualize distribution of ratings (stars)
plt.figure(figsize=(8, 6))
sns.countplot(x='stars', data=df)
plt.title('Distribution of Ratings (Stars)')
plt.xlabel('Stars')
plt.ylabel('Count')
plt.show()

# Explore relationship between stars and other features
plt.figure(figsize=(12, 6))
sns.pairplot(df[['stars', 'useful', 'funny', 'cool']])
plt.show()

# Analyze distribution of review lengths
df['review_length'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 6))
sns.histplot(df['review_length'], bins=30)
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Count')
plt.show()

# Perform sentiment analysis
# (You can use libraries like TextBlob or VADER for sentiment analysis)

# Print first few rows of preprocessed DataFrame
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['stars'], test_size=0.2, random_state=42)

# Define a pipeline for the sentiment analysis model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

