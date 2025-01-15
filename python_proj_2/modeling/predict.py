import os
from pathlib import Path

import mlflow
import pandas as pd

import typer
from loguru import logger
from tqdm import tqdm

from python_proj_2.config import MODELS_DIR, PROCESSED_DATA_DIR
from python_proj_2.some_entities_file.pipeline_params import read_pipeline_params
from python_proj_2.utils import get_sql_connection


app = typer.Typer()
os.environ['AWS_ACCESS_KEY_ID'] = 'mlflow'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5050'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_registry_uri("http://localhost:5050")

model_name = "another_model"
model_version = 'latest'
model_alias = "production"


@app.command()
def main(params):
    params = read_pipeline_params(params)
    conn = get_sql_connection(params)

    data_to_score = pd.read_sql_table(params.data_params.test_sql_tablename, conn)
    data_to_score.drop(['target'], axis=1, inplace=True)
    logger.info("Got data")

    # with open(params.train_params.model_path, 'rb') as fin:
    #     model = pickle.load(fin)

    # model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}@{model_alias}")
    logger.info("Loaded model")

    data_to_score['predict'] = model.predict_proba(data_to_score)[:, 1]
    data_to_score.to_sql(con=conn, name="batch_inference_data", if_exists='append', index=False)
    logger.info(f"Upload batch inference data")


if __name__ == "__main__":
    app()
