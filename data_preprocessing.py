import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the dataset:
    - Normalize the wage data using MinMaxScaler
    - Create time series sequences for LSTM input
    """
    # Normalize the wage data
    scaler = MinMaxScaler()
    df["Wage"] = scaler.fit_transform(df[["Wage"]])
    
    # Create sequences
    def create_sequences(data, sequence_length=12):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(df["Wage"].values)
    return X, y, scaler
