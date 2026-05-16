from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Separate features and target
X = housing.data
y = housing.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Quick checks
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# Scale features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train MLPRegressor
model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    alpha=0.001,
    learning_rate_init=0.005,
    early_stopping=True,
    random_state=42,
    max_iter=500
)
model.fit(X_train, y_train)
print(f"Training complete after {model.n_iter_} iterations.")

# Generate training predictions
train_preds = model.predict(X_train)

# Calculate train RMSE
train_rmse = mean_squared_error(y_train, train_preds) ** 0.5

print("Train RMSE:", train_rmse)

# Create Train Actual vs Predicted plot
plt.figure(figsize=(8,6))

plt.scatter(y_train, train_preds, alpha=0.5)

plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")

plt.title("Train: Actual vs Predicted")

# Perfect prediction reference line
plt.plot([0, 5], [0, 5], color='red')

# Save figure
plt.savefig("figures/train_actual_vs_pred.png")

plt.close()

# Generate test predictions
test_preds = model.predict(X_test)
test_rmse = mean_squared_error(y_test, test_preds) ** 0.5
print("Test RMSE:", test_rmse)

# Create Test Actual vs Predicted plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, test_preds, alpha=0.5)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Test: Actual vs Predicted")
plt.plot([0, 5], [0, 5], color='red')
plt.savefig("figures/test_actual_vs_pred.png")
plt.close()


# Metrics helper
def metrics_row(name, y_true, y_pred):
    return {
        "split": name,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }

metrics_df = pd.DataFrame([
    metrics_row("train", y_train, train_preds),
    metrics_row("test", y_test, test_preds),
])

print("\n=== Model Metrics ===")
print(metrics_df.to_string(index=False))

