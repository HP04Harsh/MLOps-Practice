import os
import yaml
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow


def load_params():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["train"]


def evaluate():
    if "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


    params = load_params()
    df = pd.read_csv(params["data"])
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]


    # Model load karein
    with open(params["model_path"], "rb") as f:
        model = pickle.load(f)


    preds = model.predict(X)
    acc = accuracy_score(y, preds)


    with mlflow.start_run():
        mlflow.log_metric("eval_accuracy", acc)


    print(f"[INFO] Evaluation Accuracy: {acc}")


if __name__ == "__main__":
    evaluate()
