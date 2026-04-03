import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Use only closing price
data = data[['Close']]

# Create prediction column (30 days ahead)
data['Prediction'] = data['Close'].shift(-30)

# Prepare dataset
X = data.drop(['Prediction'], axis=1)[:-30]
y = data['Prediction'][:-30]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
