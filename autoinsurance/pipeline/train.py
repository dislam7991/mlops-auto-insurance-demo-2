from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)

def make_splits(p):
    df = _read_any(p["paths"]["processed"])
    tgt = p["target"]

    # ensure target is numeric 0/1 if it's Y/N
    if df[tgt].dtype == "object":
        df[tgt] = df[tgt].map({"Y": 1, "N": 0}).fillna(df[tgt]).astype(int)

    # first split off test
    X = df.drop(columns=[tgt]); y = df[tgt]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=p["split"]["test_size"], random_state=p["split"]["random_state"], stratify=y
    )
    # split temp into train/val according to val_size over temp
    val_frac_over_temp = p["split"]["val_size"]
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac_over_temp, random_state=p["split"]["random_state"], stratify=y_temp
    )

    Path("data/splits").mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_parquet(p["paths"]["train"], index=False)
    pd.concat([X_val,   y_val],   axis=1).to_parquet(p["paths"]["val"],   index=False)
    pd.concat([X_test,  y_test],  axis=1).to_parquet(p["paths"]["test"],  index=False)

def train_and_log(p):
    # point MLflow to DagsHub (env vars also OK)
    mlflow.set_tracking_uri("https://dagshub.com/dislam7991/mlops-auto-insurance-demo.mlflow")
    mlflow.set_experiment("Insurance-ETL-Train")
    mlflow.sklearn.autolog(log_models=False)  # we log explicitly with signature+example

    # load splits
    train_df = pd.read_parquet(p["paths"]["train"])
    val_df   = pd.read_parquet(p["paths"]["val"])
    tgt = p["target"]

    y_tr = train_df[tgt]; X_tr = train_df.drop(columns=[tgt])
    y_va = val_df[tgt];   X_va = val_df.drop(columns=[tgt])

    num = X_tr.select_dtypes(include="number").columns.tolist() # numerical columns
    cat = X_tr.select_dtypes(exclude="number").columns.tolist()  # categorical columns

    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)
    ])

    clf = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    
    with mlflow.start_run(run_name="baseline-logreg") as run:
        # fit
        clf.fit(X_tr, y_tr)

        # training metrics (probabilities for ROC-AUC)
        if hasattr(clf, "predict_proba"):
            p_tr = clf.predict_proba(X_tr)[:, 1]
        else:
            # fallback: decision_function -> map to [0,1] via sigmoid
            from scipy.special import expit
            p_tr = expit(clf.decision_function(X_tr))
        yhat_tr = (p_tr >= 0.5).astype(int)

        mlflow.log_param("n_num_features", len(num))
        mlflow.log_param("n_cat_features", len(cat))
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 2000)

        mlflow.log_metric("train_accuracy", accuracy_score(y_tr, yhat_tr))
        mlflow.log_metric("train_precision", precision_score(y_tr, yhat_tr, zero_division=0))
        mlflow.log_metric("train_recall",    recall_score(y_tr, yhat_tr, zero_division=0))
        mlflow.log_metric("train_f1",        f1_score(y_tr, yhat_tr, average="binary"))
        mlflow.log_metric("train_roc_auc",   roc_auc_score(y_tr, p_tr))

        # threshold sweep on VAL to maximize F1
        thresholds = np.linspace(0.05, 0.95, 19)
        if hasattr(clf, "predict_proba"):
            p_va = clf.predict_proba(X_va)[:, 1]
        else:
            from scipy.special import expit
            p_va = expit(clf.decision_function(X_va))

        best_thr, best_f1 = 0.5, -1.0
        for t in thresholds:
            yhat = (p_va >= t).astype(int)
            f1 = f1_score(y_va, yhat, average="binary")
            mlflow.log_metric(f"val_f1_at_{t:.2f}", f1)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(t)

        mlflow.log_metric("val_f1_best", best_f1)
        mlflow.log_param("decision_threshold", best_thr)
        mlflow.log_metric("val_roc_auc", roc_auc_score(y_va, p_va))

        # log model with signature/example
        sig = infer_signature(X_tr.iloc[:5], clf.predict(X_tr.iloc[:5]))
        mlflow.sklearn.log_model(clf, "model", signature=sig, input_example=X_tr.iloc[:2].to_dict(orient="records")[0])

        # persist artifacts for downstream
        Path("models").mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, "models/model.pkl")
        with open("models/threshold.json", "w") as f:
            json.dump({"threshold": best_thr}, f)