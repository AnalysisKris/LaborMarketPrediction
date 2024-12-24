from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

def create_model(input_shape):
    """
    Build and compile the LSTM model.
    """
    model = Sequential([
        LSTM(50, activation="relu", return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation="relu", return_sequences=False),
        Dropout(0.2),
        Dense(1)  # Output layer
    ])
    
    model.compile(optimizer="adam", loss="mse")
    return model
