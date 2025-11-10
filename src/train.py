#!/usr/bin/env python3
"""
-------------------------------------------------------------------------------
This script runs the training of an XGBoost model to predict graphics card sales 
from the preprocessed data.

1. It starts by searching for the latest preprocessed CSV file in the 'data/processed/' directory.
2. If a standard model (model.pkl) does not exist, it loads the data, splits it into training and test sets, trains a model on this data, evaluates it, and then saves it as 'model/model.pkl'.
3. If a standard model already exists, it trains a new model on the latest data, evaluates it, and saves the model in the 'model/' folder in the format: model_YYYYMMDD_HHMM.pkl.
4. Performance metrics (RMSE, MAE, R²) are displayed and saved in the log file.
5. Any errors are handled and reported in the logs.

The models are saved in the 'model/' folder with the name 'model.pkl' for the standard model and with a timestamp for later versions.
The model metrics are recorded in the script’s log files.
-------------------------------------------------------------------------------
"""

import os
import sys
import pandas as pd
import xgboost as xgb
import numpy as np
import logging
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split

# Configure logging
this_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(this_dir, '../logs')
log_file_path = os.path.join(log_dir, 'train.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_latest_data_file(data_dir):
    """Finds the latest preprocessed data file."""
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No data files found in the processed directory.")
    latest_file = max(files, key=os.path.getctime)
    logging.info(f"Using latest data file: {latest_file}")
    return latest_file

def prepare_data(file_path):
    """Loads and prepares the data for training."""
    df = pd.read_csv(file_path)
    df = pd.get_dummies(df, columns=['model'], drop_first=True)
    X = df.drop('sales', axis=1)
    y = df['sales']
    return X, y

def train_and_evaluate(X, y):
    """Splits data, trains the XGBoost model, and evaluates it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Handle zero variance in the test set for R2 calculation
    if np.isclose(np.sum((y_test - y_test.mean())**2), 0):
        r2 = 0.0
    else:
        r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
    
    logging.info(f"Model Performance: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
    
    return model

def save_model(model, models_dir):
    """Saves the trained model."""
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'model.pkl')
    
    if os.path.exists(model_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_path = os.path.join(models_dir, f'model_{timestamp}.pkl')
        
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    logging.info(f"Model saved to {model_path}")

def main():
    """Main function to run the training pipeline."""
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(this_dir, '../data/processed')
        models_dir = os.path.join(this_dir, '../models')

        latest_file = get_latest_data_file(data_dir)
        X, y = prepare_data(latest_file)
        model = train_and_evaluate(X, y)
        save_model(model, models_dir)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()