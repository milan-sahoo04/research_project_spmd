import pandas as pd
import os

RAW_PATH = "data/labeled/air_quality.csv"
SAVE_PATH = "data/processed/clean_air_data.csv"

def load_data():
    print("Loading dataset...")
    return pd.read_csv(RAW_PATH)

def clean_data(df):
    print("Cleaning data...")

    # remove duplicates
    df = df.drop_duplicates()

    # remove rows with missing target
    df = df.dropna(subset=["AQI"])

    # fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=["float64","int64"]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    return df

def feature_engineering(df):
    print("Creating features...")

    # convert date if exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["hour"] = df["date"].dt.hour

    # pollution ratio feature
    if "PM2.5" in df.columns and "PM10" in df.columns:
        df["pm_ratio"] = df["PM2.5"] / df["PM10"].replace(0,1)

    return df

def save_data(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    print(f"Saved cleaned data → {SAVE_PATH}")

def main():
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    save_data(df)

if __name__ == "__main__":
    main()