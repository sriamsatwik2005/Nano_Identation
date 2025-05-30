import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("NewTrainData.csv", encoding='unicode_escape')
df = df.ffill().bfill()

# Features and target
features = ['laser pulse duration (ns)', 'laser energy (mJ)', 'loding rate (µN/s)', 'load (µN) ', 'time (s)']
target = 'Depth (nm)'
X = df[features]
y = df[target]

# Convert categorical features to string
categorical = ['laser pulse duration (ns)', 'laser energy (mJ)', 'loding rate (µN/s)']
for col in categorical:
    X[col] = X[col].astype(str)

# Convert Depth into classes
def convert_to_class(depth):
    if depth <= 100:
        return 'Shallow'
    elif depth <= 300:
        return 'Medium'
    else:
        return 'Deep'

y_class = y.apply(convert_to_class)

# Split data
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Load trained model
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# Predict numeric depth
y_pred_continuous = model.predict(X_test)

# Convert predicted depth to classes
y_pred_class = pd.Series(y_pred_continuous).apply(convert_to_class)

# Generate confusion matrix
cm = confusion_matrix(y_test_class, y_pred_class, labels=['Shallow', 'Medium', 'Deep'])

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Shallow', 'Medium', 'Deep'], yticklabels=['Shallow', 'Medium', 'Deep'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Depth Classes)')
plt.show()

# Show classification report
print("🔍 Classification Report:")
print(classification_report(y_test_class, y_pred_class))
