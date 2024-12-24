# Labor Market Inequalities Prediction with LSTM

This project uses Long Short-Term Memory (LSTM) networks to predict labor market wage trends, focusing on addressing socioeconomic inequalities, gender bias, and working-class issues within the labor market.

## Project Overview

The goal of this project is to build a predictive model using LSTM to analyze labor market data and predict future wage trends. The model incorporates data related to wages, gender, and other socioeconomic factors, aiming to support policy recommendations for reducing labor market inequalities.

## Requirements

- Python 3.7+
- TensorFlow
- pandas
- numpy
- scikit-learn
- matplotlib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/labor-market-inequalities.git
   cd labor-market-inequalities
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Model Implementation

### Data Preprocessing

The data is preprocessed as follows:

- **Normalization**: The wage data is normalized using MinMaxScaler to scale it between 0 and 1 for better model performance.
- **Sequence Creation**: The dataset is divided into time series sequences for LSTM input. A sliding window approach is used to create sequences of past data (e.g., 12 months) to predict future values.

```python
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
```

### LSTM Model

The LSTM model is built using Keras with the following architecture:

- **Input Layer**: Accepts sequences of historical labor market data.
- **LSTM Layers**: Two LSTM layers with 50 units each.
- **Dropout**: Dropout layers are added to prevent overfitting.
- **Output Layer**: A dense layer for the wage prediction.

```python
# Define LSTM model
model = Sequential([
    LSTM(50, activation="relu", return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, activation="relu", return_sequences=False),
    Dropout(0.2),
    Dense(1)  # Output layer
])

model.compile(optimizer="adam", loss="mse")
```

## Training and Evaluation

- **Model Training**: The model is trained on the preprocessed data for 50 epochs with a batch size of 32. Validation data is used to monitor performance.

```python
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
```

- **Model Evaluation**: After training, the model's performance is evaluated on the test data, and predictions are made.

```python
test_loss = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
```

---

## Results

After training the model, you can visualize the actual vs. predicted wages to evaluate performance:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(actual, label="Actual Wages")
plt.plot(predictions, label="Predicted Wages")
plt.legend()
plt.title("LSTM Wage Predictions")
plt.show()
```

---

## Extensions

The project can be extended in the following ways:

- **Incorporate More Features**: Add more socioeconomic variables like job openings, educational attainment, and regional policies.
- **Improve Model**: Use more advanced LSTM architectures like stacked LSTM, GRU, or add attention mechanisms.
- **Real-world Application**: Use actual labor market data from sources like the Bureau of Labor Statistics or OECD.
- **Model Explainability**: Integrate tools like SHAP to interpret the model’s predictions.

---

## Acknowledgments

We would like to express our sincere gratitude to the following individuals and organizations for their support and contributions to this project:

- **ChatGPT**: For providing invaluable assistance in the conceptualization, development, and coding of the project. ChatGPT helped with data preprocessing, model architecture design, and documentation writing.
- **[Your Name]**: For providing the expertise and guidance in the development of the project.
- **[Data Sources]**: For providing the labor market data used in this project, including [mention specific data sources like the Bureau of Labor Statistics, OECD, or other datasets].
- **Open-Source Community**: For the various open-source libraries and frameworks that made this project possible, including TensorFlow, Keras, Pandas, NumPy, and Matplotlib.
