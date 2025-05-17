import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your dataset
df = pd.read_csv('train.csv')  # Make sure this file is available

# Use only the 13 features that match your Streamlit input
selected_features = [
    '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GrLivArea', 'GarageArea',
    'LotArea', 'OverallQual', 'FullBath', 'BedroomAbvGr',
    'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold'
]

# Define X and y
X = df[selected_features]
y = df['SalePrice']  # Don't apply log if you’re not using it in Streamlit

# Train the model on full data
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'best_model.joblib')
