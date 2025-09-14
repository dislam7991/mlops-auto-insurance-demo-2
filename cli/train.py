import argparse, yaml
from autoinsurance.pipeline.train import train_and_log

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True)
    p = yaml.safe_load(open(ap.parse_args().params))
    train_and_log(p)