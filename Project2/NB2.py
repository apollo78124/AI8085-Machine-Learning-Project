import time
import pandas as pd
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from joblib import Parallel, delayed
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
import pickle

# Function to process a chunk of data
def process_chunk(chunk):
    X_chunk = []
    y_useful = []
    y_funny = []
    y_cool = []
    sid = SentimentIntensityAnalyzer()
    for index, row in chunk.iterrows():
        review_text = row['text'].lower()  # Convert text to lowercase
        sentiment_score = sid.polarity_scores(review_text)
        processed_features = f"{review_text} SentimentScore:{sentiment_score['compound']}"
        X_chunk.append(processed_features)
        y_useful.append(row['useful'])
        y_funny.append(row['funny'])
        y_cool.append(row['cool'])
    return X_chunk, y_useful, y_funny, y_cool


# Function to remove stop words from text
def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]
    return ' '.join(filtered_words)


# Function to add random noise to text
def add_noise(text, noise_level=0.1):
    noise_length = int(len(text) * noise_level)
    noisy_indices = random.sample(range(len(text)), noise_length)
    noisy_text = list(text)
    for idx in noisy_indices:
        noisy_text[idx] = random.choice(string.ascii_letters)
    return ''.join(noisy_text)


# Load the data in chunks and process them in parallel
data_path = "C:/Users/kunnu/Desktop/COMP 8085/Project 2/yelp_academic_dataset_review.json"
chunk_size = 5000  # Number of rows to read at a time
chunks = pd.read_json(data_path, lines=True, chunksize=chunk_size)

# Initialize lists to store processed data
X_processed = []
y_useful = []
y_funny = []
y_cool = []

# Process chunks in parallel
start_time = time.time()
for chunk in chunks:
    if len(X_processed) >= 50000:
        break  # Stop processing if we've reached the desired sample size
    for i in range(1, 6):
        subset = chunk[chunk['stars'] == i]
        if len(subset) >= 10000:
            subset = subset.sample(n=10000, random_state=42)  # Sample if enough data points are available
        else:
            subset = subset.sample(n=len(subset), random_state=42)  # Sample all available data points
        X_chunk, y_useful_chunk, y_funny_chunk, y_cool_chunk = process_chunk(subset)
        X_processed.extend(X_chunk)
        y_useful.extend(y_useful_chunk)
        y_funny.extend(y_funny_chunk)
        y_cool.extend(y_cool_chunk)
end_time = time.time()

# Convert lists to numpy arrays
X_processed = np.array(X_processed)
y_useful = np.array(y_useful)
y_funny = np.array(y_funny)
y_cool = np.array(y_cool)

# Split the processed data into training, validation, and test sets
X_train_val, X_test, y_useful_train_val, y_useful_test = train_test_split(X_processed, y_useful, test_size=0.2, random_state=42)
_, _, y_funny_train_val, y_funny_test = train_test_split(X_processed, y_funny, test_size=0.2, random_state=42)
_, _, y_cool_train_val, y_cool_test = train_test_split(X_processed, y_cool, test_size=0.2, random_state=42)
X_train, X_val, y_useful_train, y_useful_val = train_test_split(X_train_val, y_useful_train_val, test_size=0.25, random_state=42)
_, _, y_funny_train, y_funny_val = train_test_split(X_train_val, y_funny_train_val, test_size=0.25, random_state=42)
_, _, y_cool_train, y_cool_val = train_test_split(X_train_val, y_cool_train_val, test_size=0.25, random_state=42)

# Data Exploration: Distribution of 'useful', 'funny', 'cool' ratings
plt.figure(figsize=(8, 6))
sns.countplot(x=y_useful_train)
plt.title('Distribution of Useful Ratings')
plt.xlabel('Useful Rating')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x=y_funny_train)
plt.title('Distribution of Funny Ratings')
plt.xlabel('Funny Rating')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x=y_cool_train)
plt.title('Distribution of Cool Ratings')
plt.xlabel('Cool Rating')
plt.ylabel('Count')
plt.show()

# Initialize TF-IDF vectorizer for N-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)  # Adjust parameters as needed
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Naive Bayes model for 'useful'
nb_model_useful = MultinomialNB(alpha=0.1)  # Assuming best alpha is 0.1
nb_model_useful.fit(X_train_vectorized, y_useful_train)

# Make predictions on the validation set for 'useful'
y_useful_val_pred = nb_model_useful.predict(X_val_vectorized)

# Evaluate the model for 'useful'
val_accuracy_useful = accuracy_score(y_useful_val, y_useful_val_pred)
classification_rep_useful = classification_report(y_useful_val, y_useful_val_pred)

print("Evaluation for 'useful' attribute:")
print(f"Validation Accuracy: {val_accuracy_useful:.2f}")
print("Classification Report:")
print(classification_rep_useful)

# Save the 'useful' model
with open("nb_model_useful.pkl", "wb") as file:
    pickle.dump(nb_model_useful, file)

# Train the Naive Bayes model for 'funny'
nb_model_funny = MultinomialNB(alpha=0.1)  # Assuming best alpha is 0.1
nb_model_funny.fit(X_train_vectorized, y_funny_train)

# Make predictions on the validation set for 'funny'
y_funny_val_pred = nb_model_funny.predict(X_val_vectorized)

# Evaluate the model for 'funny'
val_accuracy_funny = accuracy_score(y_funny_val, y_funny_val_pred)
classification_rep_funny = classification_report(y_funny_val, y_funny_val_pred)

print("Evaluation for 'funny' attribute:")
print(f"Validation Accuracy: {val_accuracy_funny:.2f}")
print("Classification Report:")
print(classification_rep_funny)

# Save the 'funny' model
with open("nb_model_funny.pkl", "wb") as file:
    pickle.dump(nb_model_funny, file)

# Train the Naive Bayes model for 'cool'
nb_model_cool = MultinomialNB(alpha=0.1)  # Assuming best alpha is 0.1
nb_model_cool.fit(X_train_vectorized, y_cool_train)

# Make predictions on the validation set for 'cool'
y_cool_val_pred = nb_model_cool.predict(X_val_vectorized)

# Evaluate the model for 'cool'
val_accuracy_cool = accuracy_score(y_cool_val, y_cool_val_pred)
classification_rep_cool = classification_report(y_cool_val, y_cool_val_pred)

print("Evaluation for 'cool' attribute:")
print(f"Validation Accuracy: {val_accuracy_cool:.2f}")
print("Classification Report:")
print(classification_rep_cool)

# Save the 'cool' model
with open("nb_model_cool.pkl", "wb") as file:
    pickle.dump(nb_model_cool, file)

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

import pickle

# Pickle the Naive Bayes model for stars prediction
with open("nb_model_stars.pkl", "wb") as file:
    pickle.dump(nb_model, file)

# Pickle the Naive Bayes model for useful prediction
with open("nb_model_useful.pkl", "wb") as file:
    pickle.dump(nb_model_useful, file)

# Pickle the Naive Bayes model for funny prediction
with open("nb_model_funny.pkl", "wb") as file:
    pickle.dump(nb_model_funny, file)

# Pickle the Naive Bayes model for cool prediction
with open("nb_model_cool.pkl", "wb") as file:
    pickle.dump(nb_model_cool, file)
