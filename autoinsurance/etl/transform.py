from pathlib import Path
import pandas as pd

def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.lower().str.replace(" ", "_")
    # impute missing values
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def run_transform(p):
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(p["paths"]["interim"])
    df = _basic_clean(df)
    tgt = p["target"]
    if tgt in df.columns and df[tgt].dtype == "object":
        df[tgt] = df[tgt].map({"Y": 1, "N": 0}).fillna(df[tgt])
    df.to_parquet(p["paths"]["processed"], index=False)
    