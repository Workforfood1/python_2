import json
import os
import pickle
from pathlib import Path

import mlflow
import pandas as pd
import sklearn
import typer
from loguru import logger
from tqdm import tqdm

from python_proj_2.some_entities_file.pipeline_params import read_pipeline_params
from python_proj_2.config import MODELS_DIR, PROCESSED_DATA_DIR

import mlflow
from mlflow.models import infer_signature


app = typer.Typer()
# ??? {
os.environ['AWS_ACCESS_KEY_ID'] = 'YCAJE7EasWFd2LlH_j9tbt1Ar'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YCP5frOh73GPSCHB8_1OhKw7Nk259ak4wILSFhoF'
os.environ['MLFLOW_TRACKING_URI'] = 'http://89.169.171.107:8000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://storage.yandexcloud.net/'
mlflow.set_tracking_uri("http://89.169.171.107:8000")
mlflow.set_registry_uri("http://89.169.171.107:8000")
mlflow.set_experiment("workforfood1")
# }


@app.command()
def main(path):
    params = read_pipeline_params(path)

    train = pd.read_csv(params.data_params.train_data_path)
    X_train = train.drop("target", axis=1)
    y_train = train["target"].values.reshape(-1, 1)

    test = pd.read_csv(params.data_params.train_data_path)
    X_test = test.drop("target", axis=1)
    y_test = test["target"].values.reshape(-1, 1)

    model = sklearn.ensemble.LogisticRegression(n_estimators=params.train_params.n_estimators)
    model.fit(X_train, y_train)
    logger.info(f"Learn model {model}")

    y_test_pred = model.predict_proba(X_test)[:, 1]
    y_test_pred_labels = model.predict(X_test)
    roc_auc = sklearn.metrics.roc_auc_score(y_test, y_test_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred_labels)
    logger.info(f"Got ROC-AUC {roc_auc:.3f}")

    model.score(X_test, y_test)
    metrics = {"roc-auc": roc_auc, "accuracy": accuracy}
    mlflow.log_metrics(metrics)

    signature = infer_signature(X_test, model.predict(X_test))
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature,
                             registered_model_name="workforfood1-model")

    with open(params.train_params.model_path, "wb") as fin:
        pickle.dump(model, fin)
    logger.info(f"Saved model to path {params.train_params.model_path}")

    with open(params.train_params.metrics_path, "w") as fin:
        json.dump(metrics, fin)
    logger.info(f"Saved metrics to path {params.train_params.metrics_path}")


if __name__ == "__main__":
    app()
