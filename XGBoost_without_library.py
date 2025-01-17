import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt

# Load and preprocess dataset
file_path = './dataset/Clean_Dataset.csv'
df = pd.read_csv(file_path)

# Drop irrelevant columns
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('flight', axis=1)

# Encode categorical data
df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
df.stops = pd.factorize(df.stops)[0]
df = df.join(pd.get_dummies(df.airline, prefix='airline')).drop('airline', axis=1)
df = df.join(pd.get_dummies(df.source_city, prefix='source')).drop('source_city', axis=1)
df = df.join(pd.get_dummies(df.destination_city, prefix='dest')).drop('destination_city', axis=1)
df = df.join(pd.get_dummies(df.arrival_time, prefix='arrival')).drop('arrival_time', axis=1)
df = df.join(pd.get_dummies(df.departure_time, prefix='departure')).drop('departure_time', axis=1)

# Split dataset into features and target
X, y = df.drop('price', axis=1).values, df.price.values

# Manual train-test split
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Simple Gradient Boosting Regressor Implementation
class SimpleGBR:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.base_pred = None

    def fit(self, X, y):
        # Initialize with mean of y
        self.base_pred = np.mean(y)
        residual = y - self.base_pred

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            pred = tree.predict(X)
            self.models.append(tree)
            residual -= self.learning_rate * pred

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_pred)
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

class DecisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return np.mean(y)
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.mean(y)

        feature, threshold = best_split
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_indices], y[left_indices], depth + 1),
            "right": self._build_tree(X[right_indices], y[right_indices], depth + 1),
        }

    def _find_best_split(self, X, y):
        best_mse = float("inf")
        best_split = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold

                if sum(left_indices) == 0 or sum(right_indices) == 0:
                    continue

                left_y, right_y = y[left_indices], y[right_indices]
                mse = self._calculate_mse(left_y, right_y)

                if mse < best_mse:
                    best_mse = mse
                    best_split = (feature, threshold)

        return best_split

    def _calculate_mse(self, left_y, right_y):
        left_mse = np.var(left_y) * len(left_y)
        right_mse = np.var(right_y) * len(right_y)
        return (left_mse + right_mse) / (len(left_y) + len(right_y))

    def _predict_sample(self, x, tree):
        if isinstance(tree, dict):
            if x[tree["feature"]] < tree["threshold"]:
                return self._predict_sample(x, tree["left"])
            else:
                return self._predict_sample(x, tree["right"])
        return tree

# Train and evaluate the model
model = SimpleGBR(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation metrics
r2 = 1 - (sum((y_test - y_pred) ** 2) / sum((y_test - np.mean(y_test)) ** 2))
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)
rmse = math.sqrt(mse)

print("R2:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Create visualization subplots
plt.figure(figsize=(15, 10))

# 1. Predicted vs Actual Values Scatter Plot
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Values')
plt.grid(True)

# 2. Residuals Plot
plt.subplot(2, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.grid(True)

# 3. Residuals Distribution
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True)

# 4. Error Distribution Box Plot
plt.subplot(2, 2, 4)
plt.boxplot(residuals)
plt.ylabel('Prediction Error')
plt.title('Distribution of Prediction Errors')
plt.grid(True)

plt.tight_layout()
plt.show()

# Additional plot for prediction error by price range
plt.figure(figsize=(10, 6))
price_ranges = pd.qcut(y_test, q=10)
mean_errors = pd.DataFrame({'residuals': abs(residuals)}).groupby(price_ranges).mean()
plt.bar(range(len(mean_errors)), mean_errors['residuals'])
plt.xlabel('Price Range (Deciles)')
plt.ylabel('Mean Absolute Error')
plt.title('Prediction Error by Price Range')
plt.xticks(range(len(mean_errors)), ['Low', '2', '3', '4', '5', '6', '7', '8', '9', 'High'], rotation=45)
plt.grid(True)
plt.show()
