from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import struct, socket

def ip_to_numeric(ip):
    return struct.unpack("!L", socket.inet_aton(ip))[0]

# Load the training, validation, and test sets from separate CSV files
train_data = pd.read_csv('./clean_train.csv',
                         dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string",

                                }, low_memory=False)

val_data = pd.read_csv('./clean_val.csv', dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"})

# Handle non-numeric values in 'srcip' column
# Convert IP addresses to integer representation
train_data['srcip'] = train_data['srcip'].apply(ip_to_numeric)
val_data['srcip'] = val_data['srcip'].apply(ip_to_numeric)
train_data['dstip'] = train_data['dstip'].apply(ip_to_numeric)
val_data['dstip'] = val_data['dstip'].apply(ip_to_numeric)
# Drop the original 'srcip' columns

train_data['sport'] = pd.to_numeric(train_data['sport'], errors='coerce')
val_data['sport'] = pd.to_numeric(val_data['sport'], errors='coerce')
train_data['dsport'] = pd.to_numeric(train_data['dsport'], errors='coerce')
val_data['dsport'] = pd.to_numeric(val_data['dsport'], errors='coerce')

#selected_columns = ['sttl', 'ct_state_ttl', 'dttl', 'tcprtt', 'ackdat', 'synack', 'Ltime', 'Stime', 'dmeansz', 'Dload', 'state']
selected_columns = ['srcip','sport', 'dsport', 'dsport'
                    #'Stime', 'dmeansz', 'Dload', 'state',
                    #'sport', 'dsport', 'proto', 'service', 'Spkts', 'Dpkts'
                    ]
# Split the data into features and the target variable
X_train = train_data[selected_columns]  # Features for training
y_train = train_data['attack_cat']                # Target variable for training

X_val = val_data[selected_columns]      # Features for validation
y_val = val_data['attack_cat']                     # Target variable for validation

# Create an instance of the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)

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

