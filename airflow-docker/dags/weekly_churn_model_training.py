import logging
import subprocess
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import joblib
import tensorflow as tf  # Fixed typo
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

import os

# Set the necessary environment variables for DVC
os.environ['DVC_USER'] = 'limengkruy'
os.environ['DVC_PASSWORD'] = 'b1a657130ad5a3da233386baf46baa35bc361f25'  # Or use a personal access token for 2FA
# Define the DAG
dag = DAG(
    'weekly_churn_model_training',
    description='A DAG for weekly churn model training and deployment',
    schedule_interval='@weekly',
    start_date=datetime(2025, 4, 1),
    catchup=False,
)

main_path = '/opt/airflow/dags/'
# Data Preprocessing function
def load_and_preprocess_data():
    pd.set_option('display.max_columns', None)  # Show all columns

    # Disable GPU (Optional)
    tf.config.set_visible_devices([], 'GPU')

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logging.info("GPU is now enabled with Metal API")
    
    df = pd.read_excel(main_path + 'data/dataset/dataset_v01.xlsx')
    df = df.drop(columns=['customerID'], errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Check for missing values and fill with 0
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = df[['tenure', 'MonthlyCharges', 'TotalCharges']].fillna(0)
    
    # Convert SeniorCitizen to object type for encoding
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')

    # Standardization for numeric columns
    scaler = StandardScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # Label Encoding for binary columns
    label_enc_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    le = LabelEncoder()
    for col in label_enc_cols:
        df[col] = le.fit_transform(df[col])

    # One-Hot Encoding for nominal columns
    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Replace inf/-inf with NaN and fill with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0).astype(int)

    # Feature and Target Separation
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    input_example = X_train.head(1)
    
    return {
        "df" : df,
        "X_train" : X_train,
        "X_test" : X_test,
        "y_train" : y_train,
        "y_test" : y_test,
        "input_example" : input_example
    }

def train_model():
    logging.info("Starting model training...")

    prep = load_and_preprocess_data()
    models = {
        "LogisticRegression": LogisticRegression(class_weight='balanced', random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
        "KernelSVM": SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42),
        "RandomForest": RandomForestClassifier(
            class_weight='balanced', 
            n_estimators=100,
            max_depth=10,      
            random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            n_estimators=100,  
            max_depth=10,      
            learning_rate=0.1, 
            random_state=42
        )
    }

    df = prep["df"]
    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    input_example = prep["input_example"]
    
    # Train all models
    for model_name, model in models.items():
        logging.info(f"Training model: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        eval_metrics = get_model_metrics(X_test, y_test, y_pred, model, model_name)

        # Log eval_metrics to Airflow logs
        logging.info(f"Evaluation Metrics for {model_name}: {eval_metrics}")
        
        # Save the model
        joblib.dump(model, main_path + f'data/model/{model_name}.pkl')
        logging.info(f"Model {model_name} saved.")

    # ANN Model (TensorFlow)
    logging.info("Training ANN model...")
    model_ann = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train ANN for 10 epochs
    history = model_ann.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    # Predictions
    y_pred = (model_ann.predict(X_test) > 0.5).astype(int)
    
    # Log ANN metrics
    ann_metrics = get_model_metrics(X_test, y_test, y_pred, model_ann, 'ANN')
    
    # Log eval_metrics to Airflow logs
    logging.info(f"Evaluation Metrics for ANN: {ann_metrics}")
    
    # Save the ANN model
    model_ann.save(main_path + f'data/model/ANN_model.keras')
    logging.info("ANN model saved.")
        
    logging.info("Model training completed.")
    return True

def get_model_metrics(X_test, y_test, y_pred, model, model_name):
    # Calculate various evaluation metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else roc_auc_score(y_test, model.predict(X_test))

    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "model": model_name,
        "confusion_matrix_tn": TN,
        "confusion_matrix_fp": FP,
        "confusion_matrix_fn": FN,
        "confusion_matrix_tp": TP,
        "accuracy": accuracy,
        "roc_auc_score": roc_auc,
        "mse": mse,
        "r2_score": r2
    }
    
    if hasattr(model, "n_estimators"):  # RandomForest, XGBoost, etc.
        metrics["n_estimators"] = model.n_estimators
    if hasattr(model, "max_depth"):  # DecisionTree, RandomForest, XGBoost, etc.
        metrics["max_depth"] = model.max_depth
    if hasattr(model, 'learning_rate'):
        metrics["learning_rate"] = model.learning_rate

    return metrics

def push_model_to_dagshub():
    logging.info("Pushing model to DAGsHub...")

    # Push the model and data using DVC
    subprocess.run(['dvc', 'push'], cwd='/opt/airflow/dags')
    logging.info("Model pushed to DAGsHub.")

# Define the tasks
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

push_model_task = PythonOperator(
    task_id='push_model_to_dagshub',
    python_callable=push_model_to_dagshub,
    dag=dag,
)

# Set task dependencies
train_model_task >> push_model_task
