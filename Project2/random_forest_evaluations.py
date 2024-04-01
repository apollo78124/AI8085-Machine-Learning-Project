import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print("\n")
# getting evaluation scores for stars

validation_data = pd.read_json("yelp_reviews_validation_set.json")
test_data = pd.read_json("yelp_reviews_test_set.json")

X_val = validation_data['text']
y_val = validation_data['stars']

X_test = test_data['text']
y_test = test_data['stars']

vectorizer = TfidfVectorizer(max_features=5000)  

X_val_tfidf = vectorizer.fit_transform(X_val)

X_test_tfidf = vectorizer.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_val_tfidf, y_val)

test_predictions = rf_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Stars test accuracy:", test_accuracy)
classification_metrics = classification_report(y_test, test_predictions, digits=4)

print("Classification Report:")
print(classification_metrics)


# getting evaluation scores for cool

y_val = validation_data['cool']
y_test = test_data['cool']

vectorizer = TfidfVectorizer(max_features=5000)  

X_val_tfidf = vectorizer.fit_transform(X_val)

X_test_tfidf = vectorizer.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_val_tfidf, y_val)

test_predictions = rf_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Cool test accuracy:", test_accuracy)

# classification_metrics = classification_report(y_test, test_predictions, digits=4)

# print("Classification Report:")
# print(classification_metrics)

# getting evaluation scores for funny

y_val = validation_data['funny']
y_test = test_data['funny']

vectorizer = TfidfVectorizer(max_features=5000)  

X_val_tfidf = vectorizer.fit_transform(X_val)

X_test_tfidf = vectorizer.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_val_tfidf, y_val)

test_predictions = rf_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Funny test accuracy:", test_accuracy)

# classification_metrics = classification_report(y_test, test_predictions, digits=4)

# print("Classification Report:")
# print(classification_metrics)


# getting evaluation scores for useful

y_val = validation_data['useful']
y_test = test_data['useful']

vectorizer = TfidfVectorizer(max_features=5000)  

X_val_tfidf = vectorizer.fit_transform(X_val)

X_test_tfidf = vectorizer.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_val_tfidf, y_val)

test_predictions = rf_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Useful test accuracy:", test_accuracy)

# classification_metrics = classification_report(y_test, test_predictions, digits=4)

# print("Classification Report:")
# print(classification_metrics)