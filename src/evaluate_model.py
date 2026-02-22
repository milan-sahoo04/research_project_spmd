# import pandas as pd
# import joblib
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# model = joblib.load("models/aqi_model.pkl")
# scaler = joblib.load("models/scaler.pkl")

# df = pd.read_csv("data/processed/clean_air_data.csv")

# X = df.drop("AQI", axis=1)
# y = df["AQI"]

# X_scaled = scaler.transform(X)

# pred = model.predict(X_scaled)

# print("MAE:", mean_absolute_error(y,pred))
# print("RMSE:", mean_squared_error(y,pred, squared=False))
# print("R2:", r2_score(y,pred))



import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

model = joblib.load("models/aqi_model.pkl")
scaler = joblib.load("models/scaler.pkl")

df = pd.read_csv("data/processed/clean_air_data.csv")

X = df.drop("AQI", axis=1)
y = df["AQI"]

X_scaled = scaler.transform(X)
pred = model.predict(X_scaled)

print("MAE:", mean_absolute_error(y,pred))
print("RMSE:", np.sqrt(mean_squared_error(y,pred)))
print("R2:", r2_score(y,pred))