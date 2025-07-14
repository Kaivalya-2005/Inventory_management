# ml_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load
df = pd.read_csv('store_item_demand.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['store','item','date'])

# 2. Feature engineering
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['lag_7'] = df.groupby(['store','item'])['sales'].shift(7)
df['lag_14'] = df.groupby(['store','item'])['sales'].shift(14)
df = df.dropna()

X = df[['store','item','day','month','weekday','lag_7','lag_14']]
X = pd.get_dummies(X, columns=['store','item'])
y = df['sales']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# 5. Save
joblib.dump((model, X.columns.tolist()), 'xgb_inventory_model.pkl')
