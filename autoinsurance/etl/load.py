from pathlib import Path
import pandas as pd

def run_load(p):
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(p["paths"]["processed"])
    df.to_parquet("data/processed/dataset.parquet", index=False)


