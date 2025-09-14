from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Evidently 0.7+ API
from evidently import Report
from evidently.presets import DataDriftPreset

import os
uri = os.getenv("MLFLOW_TRACKING_URI")
if uri:
    mlflow.set_tracking_uri(uri)
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "auto-insurance"))


def evaluate_and_report(p):
    Path("reports").mkdir(parents=True, exist_ok=True)

    test = pd.read_parquet(p["paths"]["test"])
    tgt = p["target"]
    y = test[tgt]; X = test.drop(columns=[tgt])

    model = joblib.load("models/model.pkl")
    thr = 0.5
    try:
        with open("models/threshold.json") as f:
            thr = float(json.load(f)["threshold"])
    except Exception:
        pass  # fallback to 0.5 if not found

    # probabilities for metrics
    if hasattr(model, "predict_proba"):
        p_test = model.predict_proba(X)[:, 1]
    else:
        from scipy.special import expit
        p_test = expit(model.decision_function(X))
    yhat = (p_test >= thr).astype(int)

    acc = float(accuracy_score(y, yhat))
    pre = float(precision_score(y, yhat, zero_division=0))
    rec = float(recall_score(y, yhat, zero_division=0))
    f1  = float(f1_score(y, yhat, average="binary"))
    roc = float(roc_auc_score(y, p_test))

    # save to JSON
    with open("reports/metrics.json", "w") as f:
        json.dump(
            {"test_accuracy": acc, "test_precision": pre, "test_recall": rec, "test_f1": f1, "test_roc_auc": roc, "threshold": thr},
            f, indent=2
        )

    # confusion matrix image
    cm = confusion_matrix(y, yhat)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png"); plt.close()

    # log to MLflow
    mlflow.set_tracking_uri("https://dagshub.com/dislam7991/mlops-auto-insurance-demo.mlflow")
    mlflow.set_experiment("Insurance-ETL-Train")
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics({
            "test_accuracy": acc, "test_precision": pre, "test_recall": rec, "test_f1": f1, "test_roc_auc": roc
        })
        mlflow.log_param("decision_threshold", thr)
        mlflow.log_artifact("reports/metrics.json")
        mlflow.log_artifact("reports/confusion_matrix.png")

def drift_report(p):
    Path("reports").mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(p["paths"]["train"])
    test  = pd.read_parquet(p["paths"]["test"])
    report = Report([DataDriftPreset()])
    result = report.run(current_data=test, reference_data=train)
    result.save_html("reports/evidently_data_report.html")