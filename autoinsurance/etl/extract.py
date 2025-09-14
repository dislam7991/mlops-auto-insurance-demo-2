from pathlib import Path
import pandas as pd

def run_extract(p):
    raw = Path(p["paths"]["raw"])
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw)
    df.to_parquet(p["paths"]["interim"], index=False)