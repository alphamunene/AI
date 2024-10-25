import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_path = r'C:\Users\User\Downloads\Nairobi Office Price Ex-1.csv'
data = pd.read_csv(data_path)

x = data['SIZE'].values
y = data['PRICE'].values

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(x, y, m, c, learning_rate):
    n = len(x)
    y_pred = m * x + c
    dm = (-2/n) * np.sum(x * (y - y_pred))
    dc = (-2/n) * np.sum(y - y_pred)
    m = m - learning_rate * dm
    c = c - learning_rate * dc
    return m, c

m, c = np.random.randn(), np.random.randn()
learning_rate = 0.00001
epochs = 10

errors = []
for epoch in range(epochs):
    y_pred = m * x + c
    error = mean_squared_error(y, y_pred)
    errors.append(error)
    print(f"Epoch {epoch+1}/{epochs}, MSE: {error:.4f}")
    m, c = gradient_descent(x, y, m, c, learning_rate)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label="Actual Data")
plt.plot(x, m * x + c, color='red', label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft)")
plt.ylabel("Office Price")
plt.title("Line of Best Fit for Office Size vs Price")
plt.legend()
plt.show()

predicted_price_100 = m * 100 + c
print(f"Predicted price for 100 sq. ft: {predicted_price_100}")
