import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt

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
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            np.std(y) < 1e-6):  # Nearly pure node
            return self._create_leaf(y)

        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:  # No valid split found
            return self._create_leaf(y)
            
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # Create child nodes
        left_tree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_tree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return {'feature': best_feature,
                'threshold': best_threshold,
                'left': left_tree,
                'right': right_tree}

    def _create_leaf(self, y):
        return np.mean(y)

    def _find_best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._calculate_variance_reduction(X[:, feature], y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold

    def _calculate_variance_reduction(self, X_column, y, threshold):
        parent_var = np.var(y) * len(y)
        
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return -np.inf
        
        left_var = np.var(y[left_idxs]) * np.sum(left_idxs)
        right_var = np.var(y[right_idxs]) * np.sum(right_idxs)
        
        variance_reduction = parent_var - (left_var + right_var)
        return variance_reduction

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):
            return node
            
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])

def preprocess_data(file_path, sample_ratio=0.02, train_ratio=0.8):
    """
    Veriyi yükler, önişler ve train-test setlerine ayırır
    
    Parameters:
    -----------
    file_path: str
        CSV dosyasının yolu
    sample_ratio: float
        Kullanılacak veri oranı (0-1 arası)
    train_ratio: float
        Eğitim seti oranı (0-1 arası)
    
    Returns:
    --------
    X_train, X_test, y_train, y_test: numpy arrays
    """
    # Veriyi yükle
    df = pd.read_csv(file_path)
    
    # Gereksiz sütunları kaldır
    df = df.drop(['Unnamed: 0', 'flight'], axis=1)
    
    # Kategorik verileri kodla
    df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
    df.stops = pd.factorize(df.stops)[0]
    
    # One-hot encoding uygula
    categorical_columns = ['airline', 'source_city', 'destination_city', 
                         'arrival_time', 'departure_time']
    for col in categorical_columns:
        df = df.join(pd.get_dummies(df[col], prefix=col)).drop(col, axis=1)
    
    # Features ve target'ı ayır
    X, y = df.drop('price', axis=1).values, df.price.values
    
    # Veri setini örnekle
    if sample_ratio < 1.0:
        random_indices = np.random.permutation(len(X))
        subset_size = int(len(X) * sample_ratio)
        selected_indices = random_indices[:subset_size]
        X = X[selected_indices]
        y = y[selected_indices]
    
    # Train-test split
    split_index = int(len(X) * train_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test

def visualize_results(y_test, y_pred):
    """
    Model sonuçlarını görselleştirir
    
    Parameters:
    -----------
    y_test: numpy array
        Gerçek değerler
    y_pred: numpy array
        Tahmin edilen değerler
    """
    # Residuals hesapla
    residuals = y_test - y_pred
    
    # Ana görselleştirme
    plt.figure(figsize=(15, 10))
    
    # 1. Predicted vs Actual Values Scatter Plot
    plt.subplot(2, 2, 1)
    density = plt.hist2d(y_test, y_pred, bins=50, cmap='viridis',
                        norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(density[3], label='Nokta sayısı')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Mükemmel Tahmin')
    plt.xlabel('Gerçek Fiyat')
    plt.ylabel('Tahmin Edilen Fiyat')
    plt.title('Tahmin vs Gerçek Değerler\nRenk yoğunluğu nokta sayısını gösterir')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Residuals Plot
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Tahmin Edilen Fiyat')
    plt.ylabel('Artık Değerler')
    plt.title('Artık Değerler vs Tahminler')
    plt.grid(True)
    
    # 3. Residuals Distribution
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.xlabel('Artık Değer')
    plt.ylabel('Frekans')
    plt.title('Artık Değerlerin Dağılımı')
    plt.grid(True)
    
    # 4. Error Distribution Box Plot
    plt.subplot(2, 2, 4)
    plt.boxplot(residuals)
    plt.ylabel('Tahmin Hatası')
    plt.title('Tahmin Hatalarının Dağılımı')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Fiyat aralığına göre hata dağılımı
    plt.figure(figsize=(10, 6))
    price_ranges = pd.qcut(y_test, q=10)
    mean_errors = pd.DataFrame({'residuals': abs(residuals)}).groupby(price_ranges, observed=True).mean()
    plt.bar(range(len(mean_errors)), mean_errors['residuals'])
    plt.xlabel('Fiyat Aralığı (Ondalıklar)')
    plt.ylabel('Ortalama Mutlak Hata')
    plt.title('Fiyat Aralığına Göre Tahmin Hatası')
    plt.xticks(range(len(mean_errors)), 
              ['Düşük', '2', '3', '4', '5', '6', '7', '8', '9', 'Yüksek'], 
              rotation=45)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Veriyi hazırla
    file_path = './dataset/Clean_Dataset_.csv'
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    
    # Modeli eğit
    model = SimpleGBR(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    
    # Tahmin yap
    y_pred = model.predict(X_test)
    
    # Metrikleri hesapla
    r2 = 1 - (sum((y_test - y_pred) ** 2) / sum((y_test - np.mean(y_test)) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = math.sqrt(mse)
    
    # Sonuçları yazdır
    print("R2:", r2)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    
    # Görselleştir
    visualize_results(y_test, y_pred)
