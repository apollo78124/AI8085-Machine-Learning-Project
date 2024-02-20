from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the training, validation, and test sets from separate CSV files
train_data = pd.read_csv('./clean_train.csv', dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"})
val_data = pd.read_csv('./clean_val.csv', dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"})

selected_columns = ['sttl', 'ct_state_ttl', 'dttl', 'tcprtt', 'ct_dst_src_ltm', 'state']

# Split the data into features and the target variable
X_train = train_data[selected_columns]  # Features for training
y_train = train_data['Label']                # Target variable for training

X_val = val_data[selected_columns]      # Features for validation
y_val = val_data['Label']                     # Target variable for validation

# Create an instance of the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the validation data
y_val_pred = rf_classifier.predict(X_val)

# Evaluate the model on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Print classification report for validation set
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))