import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from xgboost import XGBRegressor

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
X, y = df.drop('price', axis=1), df.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", math.sqrt(mean_squared_error(y_test, y_pred)))

# Feature importance (optional visualization)
import matplotlib.pyplot as plt
plt.barh(X.columns, model.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in XGBoost")
plt.show()

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

# 4. Feature Importance (Top 15)
plt.subplot(2, 2, 4)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

# Plot top 15 most important features
top_n = 15
plt.barh(range(top_n), feature_importance['importance'][-top_n:])
plt.yticks(range(top_n), feature_importance['feature'][-top_n:])
plt.xlabel('Feature Importance Score')
plt.title(f'Top {top_n} Most Important Features')

plt.tight_layout()
plt.show()
