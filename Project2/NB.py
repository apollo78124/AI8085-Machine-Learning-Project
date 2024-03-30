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


# Function to process a chunk of data
def process_chunk(chunk):
    X_chunk = []
    y_chunk = []
    sid = SentimentIntensityAnalyzer()
    for index, row in chunk.iterrows():
        review_text = row['text'].lower()  # Convert text to lowercase
        sentiment_score = sid.polarity_scores(review_text)
        processed_features = f"{review_text} SentimentScore:{sentiment_score['compound']}"
        X_chunk.append(processed_features)
        y_chunk.append(row['stars'])
    return X_chunk, y_chunk


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
y_processed = []

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
        X_chunk, y_chunk = process_chunk(subset)
        X_processed.extend(X_chunk)
        y_processed.extend(y_chunk)
end_time = time.time()

# Convert lists to numpy arrays
X_processed = np.array(X_processed)
y_processed = np.array(y_processed)

# Split the processed data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Data Exploration: Distribution of Star Ratings
plt.figure(figsize=(8, 6))
sns.countplot(x='stars', data=pd.DataFrame({'stars': y_processed}))
plt.title('Distribution of Star Ratings')
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.show()

# Initialize TF-IDF vectorizer for N-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)  # Adjust parameters as needed
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Naive Bayes model
nb_model = MultinomialNB(alpha=0.1)  # Assuming best alpha is 0.1
nb_model.fit(X_train_vectorized, y_train)

# Make predictions on the validation set
y_val_pred = nb_model.predict(X_val_vectorized)
y_test_pred = nb_model.predict(X_test_vectorized)

# Evaluate the model
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
classification_rep = classification_report(y_test, y_test_pred)

# Print the evaluation metrics
print(f"Validation Accuracy with best alpha (0.1): {val_accuracy:.2f}")
print(f"Test Accuracy with best alpha (0.1): {test_accuracy:.2f}")
print("Classification Report:")
print(classification_rep)

# Save the datasets to files
np.savez("training_data.npz", X=X_train_vectorized, y=y_train)
np.savez("validation_data.npz", X=X_val_vectorized, y=y_val)
np.savez("test_data.npz", X=X_test_vectorized, y=y_test)

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

# Apply stop words removal to the processed text
X_train_processed_stopwords = [remove_stop_words(text) for text in X_train]
X_val_processed_stopwords = [remove_stop_words(text) for text in X_val]
X_test_processed_stopwords = [remove_stop_words(text) for text in X_test]

# Train the Naive Bayes model with stop words removal
vectorizer_stopwords = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X_train_vectorized_stopwords = vectorizer_stopwords.fit_transform(X_train_processed_stopwords)
X_val_vectorized_stopwords = vectorizer_stopwords.transform(X_val_processed_stopwords)

nb_model_stopwords = MultinomialNB()
nb_model_stopwords.fit(X_train_vectorized_stopwords, y_train)

# Evaluate the model with stop words removal
y_pred_stopwords = nb_model_stopwords.predict(X_val_vectorized_stopwords)
accuracy_stopwords = accuracy_score(y_val, y_pred_stopwords)

print("Experiment 1: Impact of Stop Words Removal")
print(f"Validation Accuracy with Stop Words Removal: {accuracy_stopwords:.2f}")
print("Classification Report with Stop Words Removal:")
print(classification_report(y_val, y_pred_stopwords))


# Add noise to the processed text
X_train_noisy = [add_noise(text) for text in X_train]
X_val_noisy = [add_noise(text) for text in X_val]
X_test_noisy = [add_noise(text) for text in X_test]

# Train the Naive Bayes model with noisy data
vectorizer_noisy = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X_train_vectorized_noisy = vectorizer_noisy.fit_transform(X_train_noisy)
X_val_vectorized_noisy = vectorizer_noisy.transform(X_val_noisy)

nb_model_noisy = MultinomialNB()
nb_model_noisy.fit(X_train_vectorized_noisy, y_train)

# Evaluate the model with noisy data
y_pred_noisy = nb_model_noisy.predict(X_val_vectorized_noisy)
accuracy_noisy = accuracy_score(y_val, y_pred_noisy)

print("Experiment 2: Model Robustness to Noisy Data")
print(f"Validation Accuracy with Noisy Data: {accuracy_noisy:.2f}")
print("Classification Report with Noisy Data:")
print(classification_report(y_val, y_pred_noisy))


# Load pre-trained word embeddings (GloVe)
# Convert GloVe embeddings to Word2Vec format
glove_file = "C:/Users/kunnu/Desktop/COMP 8085/Project 2/Project 2/glove.6B.100d.txt"
word2vec_output_file = "C:/Users/kunnu/Desktop/COMP 8085/Project 2/Project 2/glove.6B.100d.word2vec.txt"

# Load the GloVe word embeddings model
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Function to convert text to GloVe embeddings
def text_to_glove_embeddings(text):
    words = text.split()
    embeddings = [glove_model[word] for word in words if word in glove_model]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(glove_model.vector_size)  # Return zeros if no valid embeddings found

# Convert text to GloVe embeddings
X_train_glove_embeddings = [text_to_glove_embeddings(text) for text in X_train]
X_val_glove_embeddings = [text_to_glove_embeddings(text) for text in X_val]
X_test_glove_embeddings = [text_to_glove_embeddings(text) for text in X_test]


# Train logistic regression model with GloVe embeddings
logreg_model_glove = LogisticRegression()
logreg_model_glove.fit(X_train_glove_embeddings, y_train)

# Evaluate the model with GloVe embeddings
accuracy_glove = logreg_model_glove.score(X_val_glove_embeddings, y_val)

print("Experiment 3: Transfer Learning with Pre-trained Embeddings")
print(f"Validation Accuracy with GloVe Embeddings: {accuracy_glove:.2f}")
