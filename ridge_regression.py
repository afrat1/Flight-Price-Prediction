import pandas as pd
import numpy as np
import math

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Custom implementation of train_test_split"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]
        
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    else:
        y_train = y[train_idx]
        y_test = y[test_idx]
    
    return X_train, X_test, y_train, y_test

class StandardScaler:
    """Custom implementation of StandardScaler"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1  # Prevent division by zero
        return (X - self.mean_) / self.scale_
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_

def r2_score(y_true, y_pred):
    """Custom implementation of R2 score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def mean_squared_error(y_true, y_pred):
    """Custom implementation of MSE"""
    return np.mean((y_true - y_pred) ** 2)

class CustomRidgeRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, learning_rate=0.01):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        # Add bias term
        X = np.c_[np.ones((X.shape[0], 1)), X]
        n_samples, n_features = X.shape
        
        # Initialize weights with small random values
        self.coef_ = np.random.randn(n_features) * 0.01
        
        # Gradient descent with adaptive learning rate
        prev_loss = float('inf')
        learning_rate = self.learning_rate
        
        for iteration in range(self.max_iter):
            # Predictions
            y_pred = np.dot(X, self.coef_)
            
            # Compute gradients (with regularization)
            gradients = (-2/n_samples) * np.dot(X.T, (y - y_pred)) + \
                       (2 * self.alpha * self.coef_)
            gradients[0] -= 2 * self.alpha * self.coef_[0]  # Don't regularize intercept
            
            # Update weights with current learning rate
            new_coef = self.coef_ - learning_rate * gradients
            
            # Compute new loss
            self.coef_ = new_coef
            current_loss = self._compute_loss(X, y)
            
            # Adjust learning rate if loss is not improving
            if current_loss > prev_loss:
                learning_rate *= 0.5
            elif iteration % 10 == 0:  # Occasionally increase learning rate
                learning_rate *= 1.1
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.tol:
                break
                
            prev_loss = current_loss
        
        # Split intercept and coefficients
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    
    def _compute_loss(self, X, y):
        y_pred = np.dot(X, self.coef_)
        mse = np.mean((y - y_pred) ** 2)
        regularization = self.alpha * np.sum(self.coef_[1:] ** 2)  # Don't regularize intercept
        return mse + regularization

file_path = './dataset/Clean_Dataset.csv'

df = pd.read_csv(file_path)

df = df.drop('Unnamed: 0', axis=1)
df = df.drop('flight', axis=1)

df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)

df.stops = pd.factorize(df.stops)[0]

df = df.join(pd.get_dummies(df.airline, prefix='airline')).drop('airline', axis=1)
df = df.join(pd.get_dummies(df.source_city, prefix='source')).drop('source_city', axis=1)
df = df.join(pd.get_dummies(df.destination_city, prefix='dest')).drop('destination_city', axis=1)
df = df.join(pd.get_dummies(df.arrival_time, prefix='arrival')).drop('arrival_time', axis=1)
df = df.join(pd.get_dummies(df.departure_time, prefix='departure')).drop('departure_time', axis=1)

X, y = df.drop('price', axis=1), df.price

# Randomly sample 20% of the data
random_state = 42
sample_size = 1
indices = np.random.RandomState(random_state).permutation(len(X))
sample_size = int(len(X) * sample_size)
sample_indices = indices[:sample_size]

X = X.iloc[sample_indices]
y = y.iloc[sample_indices]

print(f"Using {len(X)} samples out of {len(df)} total samples")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Daha anlaşılır alpha değerleri
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
results = []

for alpha in alphas:
    model = CustomRidgeRegression(alpha=alpha)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results.append({
        'alpha': alpha,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'coefficient_sum': np.sum(np.abs(model.coef_))  # katsayıların toplamı
    })
    
    print(f"\nAlpha: {alpha}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Toplam Katsayı Büyüklüğü: {np.sum(np.abs(model.coef_)):.4f}")

# En iyi sonucu veren alpha değerini bul
best_result = max(results, key=lambda x: x['test_r2'])
print(f"\nEn iyi alpha: {best_result['alpha']}")
print(f"En iyi Test R2: {best_result['test_r2']:.4f}")

# Görselleştirme
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogx([r['alpha'] for r in results], 
                 [r['train_r2'] for r in results], 'b.-', label='Train R2')
    plt.semilogx([r['alpha'] for r in results], 
                 [r['test_r2'] for r in results], 'r.-', label='Test R2')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogx([r['alpha'] for r in results], 
                 [r['coefficient_sum'] for r in results], 'g.-')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Toplam Katsayı Büyüklüğü')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Matplotlib yüklü değil. Grafik oluşturulamadı.")