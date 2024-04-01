import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# getting evaluation scores for stars
print("\n")
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


print("Experiment 1: Tuning hyperparameter for stars")
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)

grid_search.fit(X_val_tfidf, y_val)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_rf_classifier = grid_search.best_estimator_
test_predictions_tuned = best_rf_classifier.predict(X_test_tfidf)
test_accuracy_tuned = accuracy_score(y_test, test_predictions_tuned)
print("Test Accuracy (Tuned Model):", test_accuracy_tuned)
print("\n")

print("Experiment 2: Getting top features for stars")
feature_importances = best_rf_classifier.feature_importances_
top_features_indices = feature_importances.argsort()[-10:][::-1]
top_features = vectorizer.get_feature_names_out()[top_features_indices]
print("Top Features:", top_features)

