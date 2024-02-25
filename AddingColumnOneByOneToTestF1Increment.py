from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

import pandas as pd
import struct, socket

def ip_to_numeric(ip):
    return struct.unpack("!L", socket.inet_aton(ip))[0]

def do_random_forest_OneColumn(train_data, val_data, columnName):
    selected_columns = [columnName]
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

    print("For Column:", columnName)

    macro_f1 = f1_score(y_val, y_val_pred, average='macro')
    print("Macro-F1 Score:", macro_f1)

    micro_f1 = f1_score(y_val, y_val_pred, average='micro')
    print("Micro-F1 Score:", micro_f1)
def do_random_forest(train_data, val_data, selected_columns):
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

    macro_f1 = f1_score(y_val, y_val_pred, average='macro')
    print("Macro-F1 Score:", macro_f1)

    micro_f1 = f1_score(y_val, y_val_pred, average='micro')
    print("Micro-F1 Score:", micro_f1)

# Load the training, validation, and test sets from separate CSV files
train_data = pd.read_csv('./clean_train.csv',
                         dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"}, low_memory=False)

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

train_data['ct_ftp_cmd'] = train_data['ct_ftp_cmd'].fillna(-1, inplace=True)
val_data['ct_ftp_cmd'] = val_data['ct_ftp_cmd'].fillna(-1, inplace=True)
train_data['ct_ftp_cmd'] = pd.to_numeric(train_data['ct_ftp_cmd'], errors='coerce')
val_data['ct_ftp_cmd'] = pd.to_numeric(val_data['ct_ftp_cmd'], errors='coerce')



print("\nColumn proto added")
columns_to_process = ['proto',
'srcip',
'sttl',
'sbytes',
'ct_state_ttl',
'smeansz',
'res_bdy_len',
'trans_depth',
'dstip',
'is_sm_ips_ports',
'ct_ftp_cmd',
'Sload',
'dbytes',
'dmeansz',
'state',
'dttl',
'Dpkts',
'sport',
'ct_src_dport_ltm',
'ct_dst_sport_ltm',
'ct_dst_src_ltm',
'Dload',
'Dintpkt',
'ct_srv_src',
'ct_dst_ltm',
'ct_srv_dst',
'dur',
'ct_src_ ltm',
'Sintpkt',
'Ltime',
'Stime',
'Spkts',
'dsport',
'swin',
'tcprtt',
'dwin',
'ackdat',
'synack',
'dloss',
'Djit',
'sloss',
'Sjit',
'ct_flw_http_mthd',
'is_ftp_login',
'stcpb',
'dtcpb',
'service']

for i in range(len(columns_to_process)):
    print("\nColumn ", columns_to_process[i], "Added")
    do_random_forest(train_data, val_data, columns_to_process[:i+1])

