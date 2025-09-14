import argparse, yaml
from autoinsurance.pipeline.train import make_splits


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True)
    p = yaml.safe_load(open(ap.parse_args().params))
    make_splits(p)

