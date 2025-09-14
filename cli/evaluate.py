import argparse, yaml
from autoinsurance.pipeline.evaluate import evaluate_and_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True)
    p = yaml.safe_load(open(ap.parse_args().params))
    evaluate_and_report(p)
    