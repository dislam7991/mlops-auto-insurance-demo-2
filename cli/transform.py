import argparse, yaml
from autoinsurance.etl.transform import run_transform

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True)
    p = yaml.safe_load(open(ap.parse_args().params))
    run_transform(p)


