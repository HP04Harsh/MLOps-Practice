import os
import yaml
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn


def load_params():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["train"]


def hyperparameter_tuning(X_train, y_train):
    rf = RandomForestClassifier()
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search


def train_model(params):
    # MLflow Remote tracking set karein (Credentials Environment se aayenge)
    if "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


    df = pd.read_csv(params["data"])
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=params["random_state"]
    )


    with mlflow.start_run():
        # Hyperparameter tuning execute karein
        grid_search = hyperparameter_tuning(X_train, y_train)
        best_model = grid_search.best_estimator_


        # Evaluation
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[INFO] Model Accuracy: {acc}")


        # Metrics and Parameters Log Karein
        mlflow.log_metric("accuracy", acc)
        for param, val in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param}", val)


        # Artifacts Log Karein (Text format me matrix & report)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")


        # Model Save Karein (Local Disk)
        os.makedirs(os.path.dirname(params["model_path"]), exist_ok=True)
        with open(params["model_path"], "wb") as f:
            pickle.dump(best_model, f)
        print(f"[INFO] Model successfully saved to: {params['model_path']}")


        # Model log karein MLflow me
        mlflow.sklearn.log_model(best_model, "model")


if __name__ == "__main__":
    params = load_params()
    train_model(params)
