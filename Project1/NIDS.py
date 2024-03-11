import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, f1_score

import pandas as pd
import struct, socket
import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def ip_to_numeric(ip):
    return struct.unpack("!L", socket.inet_aton(ip))[0]

def featureImportancePart1() :
    print("\n###############################################\n")
    print("Part 1 Feature Importance Analysis:")
    # Prepare your data
    selected_columns = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']
    # Load the training, validation, and test sets from separate CSV files
    train_df = pd.read_csv('./clean_train.csv',
                             dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"},
                             low_memory=False)

    # Convert IP addresses to integer representation
    train_df['srcip'] = train_df['srcip'].apply(ip_to_numeric)
    train_df['dstip'] = train_df['dstip'].apply(ip_to_numeric)
    # Drop the original 'srcip' columns

    train_df['sport'] = pd.to_numeric(train_df['sport'], errors='coerce')
    train_df['dsport'] = pd.to_numeric(train_df['dsport'], errors='coerce')

    train_df['ct_ftp_cmd'] = train_df['ct_ftp_cmd'].replace(' ', '-1').fillna('-1').astype(str)

    train_df.replace('-', np.nan, inplace=True)
    train_df.dropna(inplace=True)
    X = train_df[selected_columns]
    y_attack_cat = train_df['attack_cat']
    y_label = train_df['Label']

    # Initialize Random Forest classifiers
    rf_classifier_attack_cat = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_classifier_label = RandomForestClassifier(n_estimators=10, random_state=42)

    # Perform Recursive Feature Elimination (RFE) for 'Label'
    rfe_label = RFE(rf_classifier_label, n_features_to_select=10, step=10)
    rfe_label.fit(X, y_label)

    # Perform Recursive Feature Elimination (RFE) for 'attack_cat'
    rfe_attack_cat = RFE(rf_classifier_attack_cat, n_features_to_select=10, step=10)
    rfe_attack_cat.fit(X, y_attack_cat)

    # Get selected features for 'attack_cat'
    selected_features_attack_cat = X.columns[rfe_attack_cat.support_]

    # Get selected features for 'Label'
    selected_features_label = X.columns[rfe_label.support_]

    # Print selected features for 'attack_cat'
    print("Selected features for 'attack_cat':")
    print(selected_features_attack_cat)

    # Print selected features for 'Label'
    print("Selected features for 'Label':")
    print(selected_features_label)

    # Train the models using selected features
    rf_classifier_attack_cat.fit(X[selected_features_attack_cat], y_attack_cat)
    rf_classifier_label.fit(X[selected_features_label], y_label)

    # Print feature rankings for 'attack_cat'
    print("Feature rankings for 'attack_cat':")
    print(rfe_attack_cat.ranking_)

    # Print feature rankings for 'Label'
    print("Feature rankings for 'Label':")
    print(rfe_label.ranking_)

def covariance_Part1(data) :
    print("\n###############################################\n")
    print("Part 1 Covariance Analysis:")
    correlation_matrix = data.corr()
    correlation_with_label = correlation_matrix['Label'].sort_values(ascending=False)
    print("Correlation with 'Label' column:")
    print(correlation_with_label)

    correlation_matrix = data.corr()
    correlation_with_label = correlation_matrix['attack_cat'].sort_values(ascending=False)
    print("Correlation with 'attack_cat' column:")
    print(correlation_with_label)
def do_RandomForest_Part2(train_data, val_data, selected_columns):
    print("\n###############################################\n")
    print("Part 2 RandomForest Classification:")
    # Split the data into features and the target variable
    X_train = train_data[selected_columns]  # Features for training
    y_train = train_data['Label']  # Target variable for training

    X_val = val_data[selected_columns]  # Features for validation
    y_val = val_data['Label']  # Target variable for validation

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

def do_KNN_Part2(train_data, val_data, selected_columns):
    print("\n###############################################\n")
    print("Part 2 KNN Classification:")
    # Split the data into features and the target variable
    X_train = train_data[selected_columns]  # Features for training
    y_train = train_data['Label']  # Target variable for training

    X_val = val_data[selected_columns]  # Features for validation
    y_val = val_data['Label']  # Target variable for validation

    # Create an instance of the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=10)

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

def do_KNN_Part3(train_data, val_data, selected_columns):
    print("\n###############################################\n")
    print("Part 3 KNN Classification:")
    # Split the data into features and the target variable
    X_train = train_data[selected_columns]  # Features for training
    y_train = train_data['attack_cat']  # Target variable for training

    X_val = val_data[selected_columns]  # Features for validation
    y_val = val_data['attack_cat']  # Target variable for validation

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

    micro_f1 = f1_score(y_val, y_val_pred, average='micro')
    print("Micro-F1 Score:", micro_f1)

def do_random_forestPart3(train_data, val_data, selected_columns):
    print("\n###############################################\n")
    print("Part 3 Randomforest Classification:")
    # Split the data into features and the target variable
    X_train = train_data[selected_columns]  # Features for training
    y_train = train_data['attack_cat']                # Target variable for training

    X_val = val_data[selected_columns]      # Features for validation
    y_val = val_data['attack_cat']                     # Target variable for validation
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_val = scaler.fit_transform(X_val)
    # Create an instance of the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, criterion = 'gini', random_state=42)

    # Train the classifier on the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the validation data
    y_val_pred = rf_classifier.predict(X_val)

    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    macro_f1 = f1_score(y_val, y_val_pred, average='macro')
    print("Macro-F1 Score:", macro_f1)

    micro_f1 = f1_score(y_val, y_val_pred, average='micro')
    print("Micro-F1 Score:", micro_f1)

warnings.filterwarnings("ignore")
# Load the training, validation, and test sets from separate CSV files
train_data = pd.read_csv('./clean_train.csv',
                         dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"}, low_memory=False)

val_data = pd.read_csv('./clean_val.csv', dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"}, low_memory=False)

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

train_data['ct_ftp_cmd'] = train_data['ct_ftp_cmd'].replace(' ', '-1').fillna('-1').astype(str)
val_data['ct_ftp_cmd'] = val_data['ct_ftp_cmd'].replace(' ', '-1').fillna('-1').astype(str)

print("Project 1 Result Start\n")
featureImportancePart1()

covariance_Part1(train_data)
#Selected Features For Part 2 RandomForest
selected_columns = ['sttl', 'ct_state_ttl', 'dttl', 'tcprtt', 'ct_dst_src_ltm', 'state']
do_RandomForest_Part2(train_data, val_data, selected_columns)

#Selected Features For Part 2 KNN
selected_columns = ['sttl', 'ct_state_ttl', 'dttl', 'tcprtt', 'ct_dst_src_ltm', 'state', 'Dpkts']
do_KNN_Part2(train_data, val_data, selected_columns)

#Selected Features For Random Forest
selected_columns = ['srcip','dstip','dsport','proto','sbytes','dbytes','sttl','dttl','service','Dload','Dpkts','smeansz','dmeansz','tcprtt','synack','ackdat','ct_state_ttl','ct_dst_sport_ltm']
do_random_forestPart3(train_data, val_data, selected_columns)

# Replace invalid values with NaN
train_data.replace('-', np.nan, inplace=True)
val_data.replace('-', np.nan, inplace=True)
train_data.dropna(inplace=True)
val_data.dropna(inplace=True)
#Selected Features For KNN
selected_columns = ['sttl', 'ct_state_ttl', 'dstip', 'dsport', 'dttl', 'srcip', 'sbytes', 'tcprtt', 'ct_dst_src_ltm', 'state', 'Dpkts', 'ct_ftp_cmd', 'Dload', 'Label']
do_KNN_Part3(train_data, val_data, selected_columns)
print("\n###############################################")
print("\nEnd Of Project 1 Results")
print("Group Eunhak, Kartik, Nam")
print("###############################################")

