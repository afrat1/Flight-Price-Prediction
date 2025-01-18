import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt

class SimpleGBR:
    """
    Simple Gradient Boosting Regressor implementation
    Similar to XGBoost but with basic functionality
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Initialize the Gradient Boosting Regressor
        
        Parameters:
        -----------
        n_estimators: int
            Number of boosting stages (trees) to perform
        learning_rate: float
            Step size shrinkage used to prevent overfitting
        max_depth: int
            Maximum depth of individual regression trees
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.base_pred = None

    def fit(self, X, y):
        """
        Fit the gradient boosting model
        
        Parameters:
        -----------
        X: array-like
            Training data features
        y: array-like
            Target values
        """
        # Initialize prediction with mean of target values
        self.base_pred = np.mean(y)
        residual = y - self.base_pred

        # Iteratively train trees on residuals
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            pred = tree.predict(X)
            self.models.append(tree)
            # Update residuals based on current prediction
            residual -= self.learning_rate * pred

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_pred)
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

class DecisionTreeRegressor:
    """
    Custom Decision Tree Regressor implementation
    Used as base learner in the Gradient Boosting model
    """
    
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        """
        Recursively grows the decision tree
        
        Parameters:
        -----------
        X: array-like
            Training data
        y: array-like
            Target values
        depth: int
            Current depth in the tree
        
        Returns:
        --------
        dict or float
            Either a decision node (dict) or leaf value (float)
        """
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            np.std(y) < 1e-6):  # Nearly pure node
            return self._create_leaf(y)

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:  # No valid split found
            return self._create_leaf(y)
            
        # Split data based on best feature and threshold
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # Recursively build left and right subtrees
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
        """
        Calculate variance reduction for a potential split
        This is the splitting criterion used for regression trees
        
        Returns:
        --------
        float
            The reduction in variance achieved by this split
        """
        # Calculate parent node variance
        parent_var = np.var(y) * len(y)
        
        # Split data
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        
        # Check if split is valid
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return -np.inf
        
        # Calculate variance for children
        left_var = np.var(y[left_idxs]) * np.sum(left_idxs)
        right_var = np.var(y[right_idxs]) * np.sum(right_idxs)
        
        # Calculate variance reduction
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

def preprocess_data(file_path, sample_ratio=0.01, train_ratio=0.8):
    """
    Loads, preprocesses and splits data into train-test sets
    
    Parameters:
    -----------
    file_path: str
        Path to CSV file
    sample_ratio: float
        Ratio of data to use (between 0-1)
    train_ratio: float
        Training set ratio (between 0-1)
    
    Returns:
    --------
    X_train, X_test, y_train, y_test: numpy arrays
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Remove unnecessary columns
    df = df.drop(['Unnamed: 0', 'flight'], axis=1)
    
    # Encode categorical data
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
    Visualizes model results
    
    Parameters:
    -----------
    y_test: numpy array
        Actual values
    y_pred: numpy array
        Predicted values
    """
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Main visualization
    plt.figure(figsize=(15, 10))
    
    # 1. Predicted vs Actual Values Scatter Plot
    plt.subplot(2, 2, 1)
    density = plt.hist2d(y_test, y_pred, bins=50, cmap='viridis',
                        norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(density[3], label='Number of points')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Prediction vs Actual Values\nColor density shows number of points')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Residuals Plot
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predictions')
    plt.grid(True)
    
    # 3. Residuals Distribution
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.xlabel('Residual')
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
    
    # Error distribution by price range
    plt.figure(figsize=(10, 6))
    price_ranges = pd.qcut(y_test, q=10)
    mean_errors = pd.DataFrame({'residuals': abs(residuals)}).groupby(price_ranges, observed=True).mean()
    plt.bar(range(len(mean_errors)), mean_errors['residuals'])
    plt.xlabel('Price Range (Deciles)')
    plt.ylabel('Mean Absolute Error')
    plt.title('Prediction Error by Price Range')
    plt.xticks(range(len(mean_errors)), 
              ['Low', '2', '3', '4', '5', '6', '7', '8', '9', 'High'], 
              rotation=45)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load and preprocess the data
    file_path = './dataset/Clean_Dataset_.csv'
    X_train, X_test, y_train, y_test = preprocess_data(
        file_path, 
        sample_ratio=0.01,  # Use full dataset
        train_ratio=0.8  # 80% for training, 20% for testing
    )
    
    # Initialize and train the model
    model = SimpleGBR(
        n_estimators=120,    # Number of trees
        learning_rate=0.1,   # Learning rate for gradient descent
        max_depth=3         # Maximum depth of each tree
    )
    model.fit(X_train, y_train)
    
    # Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    r2 = 1 - (sum((y_test - y_pred) ** 2) / sum((y_test - np.mean(y_test)) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = math.sqrt(mse)
    
    print("R2:", r2)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    
    visualize_results(y_test, y_pred)
