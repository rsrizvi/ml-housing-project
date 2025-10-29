import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from create_model import DEMOGRAPHICS_PATH, RANDOM_SEED, SALES_COLUMN_SELECTION, SALES_PATH, load_data


X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED)

with open("model/model.pkl", 'rb') as f:
    model = pickle.load(f)
with open("model/model_features.json", 'r') as f:
    features = json.load(f)

preds_train = model.predict(X_train[features])
preds_test = model.predict(X_test[features])

print("Train Metrics:")
print(f"MAE: {mean_absolute_error(y_train, preds_train)}")
print(f"RMSE: {mean_squared_error(y_train, preds_train, squared=False)}")
print(f"R2: {r2_score(y_train, preds_train)}")

print("Test Metrics:")
print(f"MAE: {mean_absolute_error(y_test, preds_test)}")
print(f"RMSE: {mean_squared_error(y_test, preds_test, squared=False)}")
print(f"R2: {r2_score(y_test, preds_test)}")
