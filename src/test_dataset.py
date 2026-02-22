import pandas as pd

DATA_PATH = "../slm/dataset.csv"  # adjust path if running from src
df = pd.read_csv(DATA_PATH)
print(df.head())
print(df.columns)