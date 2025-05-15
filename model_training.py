# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt # Import sqrt

from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (assuming this has been loaded in previous cells)
# Ensure 'df' is available from previous steps, which loads 'train.csv'
# If running this cell standalone, uncomment the line below to load the data
# df = pd.read_csv('train.csv')

# Separate features (X) and target (y)
# Assuming 'SalePrice' is the target variable
if 'SalePrice' in df.columns:
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
else:
    # Handle the case where 'SalePrice' is not in the DataFrame
    # This might happen if the 'train.csv' file doesn't contain this column,
    # or if a different dataset is intended.
    # For this fix, we'll raise an error or handle appropriately.
    # For demonstration, let's assume 'SalePrice' exists.
    # If not, you would need to define X and y based on your actual data columns.
    raise ValueError("Column 'SalePrice' not found in the DataFrame.")

# Handle categorical features and missing values before splitting
# A simple approach: drop non-numeric columns and fill remaining NaNs for demonstration
# In a real scenario, you would use more sophisticated preprocessing like one-hot encoding
X = X.select_dtypes(include=np.number).fillna(X.select_dtypes(include=np.number).mean())
y = y.fillna(y.mean()) # Fill missing values in target if any

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# Store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Check for and handle potential issues with training data (e.g., infinite values)
    if np.any(np.isinf(X_train)):
        print(f"Warning: Infinite values found in X_train for model {name}. Handling...")
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        # A more robust strategy would be to handle NaNs appropriately
        # For simplicity, we'll drop rows with NaNs introduced by inf replacement
        X_train = X_train.dropna()
        y_train = y_train[X_train.index] # Align y_train with X_train after dropping rows

    # Check if training data is empty after cleaning
    if X_train.empty:
         print(f"Error: Training data is empty for model {name} after handling infinities/NaNs. Skipping.")
         continue

    model.fit(X_train, y_train)

    # Check for and handle potential issues with testing data
    if np.any(np.isinf(X_test)):
        print(f"Warning: Infinite values found in X_test for model {name}. Handling...")
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        # Align X_test and y_test after dropping NaNs
        original_test_indices = X_test.index
        X_test = X_test.dropna()
        y_test_aligned = y_test[X_test.index] # Create an aligned y_test

        if X_test.empty:
             print(f"Error: Testing data is empty for model {name} after handling infinities/NaNs. Skipping.")
             continue

        y_pred = model.predict(X_test)
        # Calculate metrics using the aligned y_test
        mse = mean_squared_error(y_test_aligned, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_test_aligned, y_pred)

    else:
        y_pred = model.predict(X_test)
        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        # Calculate Root Mean Squared Error by taking the square root of MSE
        rmse = sqrt(mse)
        r2 = r2_score(y_test, y_pred)                             # R² Score


    results[name] = {'RMSE': round(rmse, 2), 'R²': round(r2, 4)}

# Display comparison table
if results:
    results_df = pd.DataFrame(results).T.sort_values(by="RMSE")
    print("Model Performance Comparison:\n")
    print(results_df)
else:
    print("No model evaluation results to display.")