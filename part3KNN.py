from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pandas as pd
import numpy as np
import socket

# Load the training, validation, and test sets from separate CSV files
train_data = pd.read_csv('./clean_train.csv', dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"})
val_data = pd.read_csv('./clean_val.csv', dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"})

selected_columns = ['sttl', 'ct_state_ttl', 'dstip', 'dsport', 'dttl', 'srcip', 'sbytes', 'tcprtt', 'ct_dst_src_ltm', 'state', 'Dpkts', 'ct_ftp_cmd', 'Dload', 'Label']

# Convert ct_ftp_cmd to string and fill missing values with '-1'
train_data['ct_ftp_cmd'] = train_data['ct_ftp_cmd'].replace(' ', '-1').fillna('-1').astype(str)
val_data['ct_ftp_cmd'] = val_data['ct_ftp_cmd'].replace(' ', '-1').fillna('-1').astype(str)

# Function to convert IP address to integer
def ip_to_int(ip):
    try:
        return int(socket.inet_aton(ip).hex(), 16)
    except socket.error:
        return None

# Function to convert IP address to integer
def ip_to_int(ip):
    try:
        return int(socket.inet_aton(ip).hex(), 16)
    except (socket.error, ValueError):
        return None

# Function to convert port number to integer
def port_to_int(port):
    try:
        return int(port)
    except ValueError:
        return None

# Convert IP address columns to integers
ip_columns = ['srcip', 'dstip']
for col in ip_columns:
    train_data[col] = train_data[col].apply(ip_to_int)
    val_data[col] = val_data[col].apply(ip_to_int)

# Convert port number columns to integers
port_columns = ['dsport', 'sport']
for col in port_columns:
    train_data[col] = train_data[col].apply(port_to_int)
    val_data[col] = val_data[col].apply(port_to_int)

# Replace invalid values with NaN
train_data.replace('-', np.nan, inplace=True)
val_data.replace('-', np.nan, inplace=True)

# Drop rows with NaN values
train_data.dropna(inplace=True)
val_data.dropna(inplace=True)


# Further processing and model training...

data_types = train_data.dtypes

# Print the data types
print(data_types)
# Split the data into features and the target variable
X_train = train_data[selected_columns]  # Features for training
y_train = train_data['attack_cat']      # Target variable for training

X_val = val_data[selected_columns]      # Features for validation
y_val = val_data['attack_cat']          # Target variable for validation

# Create an instance of the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=20)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the validation data
y_val_pred = knn_classifier.predict(X_val)

# Evaluate the model on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Print classification report for validation set
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

macro_f1 = f1_score(y_val, y_val_pred, average='macro')
print("Macro-F1 Score:", macro_f1)
