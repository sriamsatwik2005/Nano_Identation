import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("NewTrainData.csv", encoding='unicode_escape')
df = df.ffill().bfill()

# Features and target
features = ['laser pulse duration (ns)', 'laser energy (mJ)', 'loding rate (µN/s)', 'load (µN) ', 'time (s)']
target = 'Depth (nm)'

X = df[features]
y = df[target]

# Convert selected columns to string for CatBoost categorical handling
categorical = ['laser pulse duration (ns)', 'laser energy (mJ)', 'loding rate (µN/s)']
for col in categorical:
    X.loc[:, col] = X[col].astype(str)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = CatBoostRegressor(iterations=400, learning_rate=0.2, depth=6, loss_function='RMSE', verbose=100)
model.fit(X_train, y_train, cat_features=categorical)

# Save model
model.save_model("catboost_model.cbm")
print("✅ Model trained and saved successfully.")
