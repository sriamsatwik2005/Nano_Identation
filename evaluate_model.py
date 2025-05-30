import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("NewTrainData.csv", encoding='unicode_escape')
df = df.ffill().bfill()

# Define features and target
features = ['laser pulse duration (ns)', 'laser energy (mJ)', 'loding rate (µN/s)', 'load (µN) ', 'time (s)']
target = 'Depth (nm)'

X = df[features]
y = df[target]

# Convert categorical features to string
categorical = ['laser pulse duration (ns)', 'laser energy (mJ)', 'loding rate (µN/s)']
for col in categorical:
    X[col] = X[col].astype(str)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# Predict on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print("📊 Model Evaluation Metrics:")
print(f"✅ R² Score       : {r2:.4f}")
print(f"✅ MAE (μm)       : {mae:.4f}")
print(f"✅ RMSE (μm)      : {rmse:.4f}")
