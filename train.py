import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, preprocess_data
from model import create_model

def main():
    # Load the data
    df = load_data('data/labor_market_data.csv')

    # Preprocess the data
    X, y, scaler = preprocess_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build the model
    model = create_model(input_shape=(X_train.shape[1], 1))

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')

    # Make predictions
    predictions = model.predict(X_test)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Wages")
    plt.plot(predictions, label="Predicted Wages")
    plt.legend()
    plt.title("LSTM Wage Predictions")
    plt.show()

if __name__ == "__main__":
    main()
