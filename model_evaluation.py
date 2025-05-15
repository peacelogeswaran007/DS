import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor

# Sample fallback initialization (only needed for testing; remove if you already have data)
try:
    X_train
    y_train
except NameError:
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    X, y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
best_model = GradientBoostingRegressor()
best_model.fit(X_train, y_train)

# Predict
y_pred = best_model.predict(X_test)

# Plot: Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

# Plot: Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color='salmon')
plt.title("Residuals Distribution")
plt.xlabel("Residual Error")
plt.grid(True)
plt.show()