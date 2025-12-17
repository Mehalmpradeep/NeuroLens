import pandas as pd

BASE = "D:/NeuroLens/datasets/"
CSV_PATH = "D:/NeuroLens/csv/all_train.csv"

df = pd.read_csv(CSV_PATH)

df["path"] = BASE + df["path"].astype(str)

df.to_csv(CSV_PATH, index=False)

print("✅ Paths prepended successfully")