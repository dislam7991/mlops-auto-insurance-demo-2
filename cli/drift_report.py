import argparse, yaml
from autoinsurance.pipeline.evaluate import drift_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True)
    p = yaml.safe_load(open(ap.parse_args().params))
    drift_report(p)