import pandas as pd
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

file_path = './dataset/Noisy_Dataset.csv'

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

X, y = df.drop('price', axis = 1), df.price

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

reg = RandomForestRegressor(n_jobs=-1)

reg.fit(X_train, y_train)

print(reg.score(X_test, y_test))

y_pred = reg.predict(X_test) 
print ('R2: ',r2_score(y_test,y_pred))
print ('MAE: ',mean_absolute_error(y_test,y_pred))
print ('MSE: ',mean_squared_error(y_test,y_pred))
print ('RMSE: ',math.sqrt(mean_squared_error(y_test,y_pred)))

importances = dict(zip(reg.feature_names_in_, reg.feature_importances_))
sorted_importances = sorted(importances.items(), key=lambda x:x[1], reverse=True)
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': [1.0, 'sqrt', 'auto']
}

reg = RandomForestRegressor(n_jobs=-1)

random_search = RandomizedSearchCV(estimator=reg, param_distributions=param_dist, n_iter = 2, cv = 3, scoring='neg_mean_squared_error', verbose=2, random_state=10,n_jobs=-1)
random_search.fit(X_train, y_train)

best_regressor = random_search.best_estimator_ 

print(best_regressor.score(X_test,y_test))

y_pred = best_regressor.predict(X_test) 
print ('R2: ',r2_score(y_test,y_pred))
print ('MAE: ',mean_absolute_error(y_test,y_pred))
print ('MSE: ',mean_squared_error(y_test,y_pred))
print ('RMSE: ',math.sqrt(mean_squared_error(y_test,y_pred)))