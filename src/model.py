from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

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