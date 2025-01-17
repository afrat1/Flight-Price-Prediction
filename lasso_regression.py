import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class CustomLassoRegression:
    def __init__(self, alpha=0.01, max_iter=3000, tol=1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0
        
    def soft_threshold(self, x, lambda_):
        """Soft thresholding operator"""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        
        # Calculate learning rate (step size)
        step_size = 1.0 / (np.linalg.norm(X.T @ X, 2) + 1e-8)
        
        # Center the target variable
        y_mean = np.mean(y)
        y_centered = y - y_mean
        
        for iter in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            # Compute residuals
            residuals = X @ self.coef_ - y_centered
            
            # Update coefficients using coordinate descent
            for j in range(n_features):
                r_j = residuals + X[:, j] * self.coef_[j]
                grad = X[:, j] @ r_j
                
                # Update coefficient using soft thresholding
                self.coef_[j] = self.soft_threshold(
                    self.coef_[j] - step_size * grad,
                    self.alpha * step_size
                )
                
                # Update residuals
                residuals = r_j - X[:, j] * self.coef_[j]
            
            # Check convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                print(f"Converged after {iter + 1} iterations")
                break
        
        # Store intercept
        self.intercept_ = y_mean
                
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

# Load and preprocess data
file_path = './dataset/Clean_Dataset.csv'
df = pd.read_csv(file_path)

# Take a smaller sample for faster testing
df_reduced = df.sample(frac=0.2, random_state=42)
print(f"Original dataset size: {len(df)} samples")
print(f"Reduced dataset size: {len(df_reduced)} samples")

# Data preprocessing
df_reduced = df_reduced.drop(['Unnamed: 0', 'flight'], axis=1)
df_reduced['class'] = df_reduced['class'].apply(lambda x: 1 if x == 'Business' else 0)
df_reduced.stops = pd.factorize(df_reduced.stops)[0]

# One-hot encoding
df_reduced = df_reduced.join(pd.get_dummies(df_reduced.airline, prefix='airline')).drop('airline', axis=1)
df_reduced = df_reduced.join(pd.get_dummies(df_reduced.source_city, prefix='source')).drop('source_city', axis=1)
df_reduced = df_reduced.join(pd.get_dummies(df_reduced.destination_city, prefix='dest')).drop('destination_city', axis=1)
df_reduced = df_reduced.join(pd.get_dummies(df_reduced.arrival_time, prefix='arrival')).drop('arrival_time', axis=1)
df_reduced = df_reduced.join(pd.get_dummies(df_reduced.departure_time, prefix='departure')).drop('departure_time', axis=1)

# Split features and target
X, y = df_reduced.drop('price', axis=1), df_reduced.price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different alpha values
alphas = [0.0001, 0.001, 0.01, 0.1]
results = []

for alpha in alphas:
    # Custom Lasso Regression
    print(f"\nTraining Custom Lasso Regression with alpha={alpha}...")
    custom_lasso = CustomLassoRegression(alpha=alpha, max_iter=3000, tol=1e-6)
    custom_lasso.fit(X_train_scaled, y_train)

    # Make predictions and evaluate
    y_pred_lasso = custom_lasso.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_lasso)
    mae = mean_absolute_error(y_test, y_pred_lasso)
    mse = mean_squared_error(y_test, y_pred_lasso)
    rmse = math.sqrt(mse)
    
    results.append({
        'alpha': alpha,
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse
    })
    
    print(f'Alpha: {alpha}')
    print('R2: ', r2)
    print('MAE: ', mae)
    print('MSE: ', mse)
    print('RMSE: ', rmse)

# Find best alpha
best_result = max(results, key=lambda x: x['R2'])
print("\nBest Results:")
print(f"Alpha: {best_result['alpha']}")
print(f"R2: {best_result['R2']}")
print(f"MAE: {best_result['MAE']}")
print(f"RMSE: {best_result['RMSE']}")

# Use best alpha for final model and feature importance
best_lasso = CustomLassoRegression(alpha=best_result['alpha'], max_iter=3000, tol=1e-6)
best_lasso.fit(X_train_scaled, y_train)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_lasso.coef_
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Count number of features used (non-zero coefficients)
n_features_used = np.sum(best_lasso.coef_ != 0)
print(f"\nNumber of features used by Custom Lasso: {n_features_used} out of {len(X.columns)}") 