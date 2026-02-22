# import pandas as pd
# import joblib
# import os
# import numpy as np

# from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler

# DATA_PATH = "data/processed/clean_air_data.csv"

# def load_data():
#     print("Loading processed dataset...")
#     return pd.read_csv(DATA_PATH)

# def split_data(df):

#     X = df.drop("AQI", axis=1)
#     y = df["AQI"]

#     return train_test_split(X, y, test_size=0.2, random_state=42)

# def scale_data(X_train, X_test):
#     scaler = StandardScaler()

#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     return X_train_scaled, X_test_scaled, scaler

# def train_model(X_train, y_train):
#     print("Training model...")
    
#     model = LinearRegression()

#     # model = RandomForestRegressor(
#     #     n_estimators=200,
#     #     max_depth=12,
#     #     random_state=42,
#     #     n_jobs=-1
#     # )

#     model.fit(X_train, y_train)
#     return model


# def evaluate(model, X_test, y_test):
#     from sklearn.metrics import mean_squared_error, r2_score
#     import numpy as np

#     preds = model.predict(X_test)

#     rmse = np.sqrt(mean_squared_error(y_test, preds))
#     r2 = r2_score(y_test, preds)

#     print("\nModel Performance")
#     print("RMSE:", rmse)
#     print("R2 Score:", r2)

# def save_model(model, scaler):

#     os.makedirs("models", exist_ok=True)

#     joblib.dump(model, "models/aqi_model.pkl")
#     joblib.dump(scaler, "models/scaler.pkl")

#     print("Models saved in /models folder")

# def main():

#     df = load_data()

#     X_train, X_test, y_train, y_test = split_data(df)

#     X_train, X_test, scaler = scale_data(X_train, X_test)

#     model = train_model(X_train, y_train)

#     evaluate(model, X_test, y_test)

#     save_model(model, scaler)

# if __name__ == "__main__":
#     main()



# src/train_prediction.py
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "data/processed/clean_air_data.csv"

def load_data():
    print("Loading processed dataset...")
    return pd.read_csv(DATA_PATH)

def split_data(df):
    X = df.drop("AQI", axis=1)
    y = df["AQI"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\nModel Performance:")
    print("RMSE:", round(rmse,2))
    print("MAE:", round(mae,2))
    print("R²:", round(r2,2))

def save_model(model, scaler):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/aqi_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Models saved in /models folder")

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    evaluate(model, X_test_scaled, y_test)
    save_model(model, scaler)

if __name__ == "__main__":
    main()