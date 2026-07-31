# MLOps Pipeline - Pima Indian Diabetes Prediction

An end-to-end MLOps project that builds a Random Forest classifier to predict diabetes onset from the Pima Indians Diabetes dataset. The project demonstrates a reproducible ML pipeline using **DVC** for data/pipeline versioning and **MLflow** for experiment tracking, with **DagsHub** as the remote for both.

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| scikit-learn | Model training (`RandomForestClassifier` + `GridSearchCV`) |
| DVC | Data versioning + pipeline orchestration |
| MLflow | Experiment tracking, metrics, params & artifact logging |
| DagsHub | Remote storage (DVC) + hosted MLflow tracking server |
| Git / GitHub | Source control |

## Features

- Reproducible 3-stage DVC pipeline: `pre_process` -> `train` -> `evaluate`
- Hyperparameter tuning via `GridSearchCV` (n_estimators, max_depth, min_samples_split)
- MLflow logging: accuracy, best hyperparameters, confusion matrix, classification report, serialized model
- Data and models versioned and pushed to DagsHub DVC storage
- All paths/parameters centralized in `params.yaml`

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── data.csv              # Raw dataset (tracked by DVC)
│   └── preprocessed/
│       └── data.csv              # Preprocessed output (DVC output)
├── models/
│   └── model.pkl                 # Trained model artifact (DVC output)
├── src/
│   ├── __init__.py
│   ├── pre_process.py            # Stage 1: load & save data
│   ├── train.py                  # Stage 2: tune, train, log to MLflow
│   └── evaluate.py               # Stage 3: evaluate & log metric
├── params.yaml                   # Central config (paths + hyperparams)
├── requirements.txt
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC pipeline lockfile
└── README.md
```

## Dataset

[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) - 768 records, 8 medical predictor variables and one binary target (`Outcome`):

| Column | Description |
|--------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age (years) |
| Outcome | 1 = diabetic, 0 = non-diabetic |

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset at data/raw/data.csv
```

## DagsHub Credentials

The project logs experiments and stores data on DagsHub (`https://dagshub.com/Harsh05/MLOps-Lec1`).

```bash
# Windows (cmd)
set MLFLOW_TRACKING_URI=https://dagshub.com/Harsh05/MLOps-Lec1.mlflow
set MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
set MLFLOW_TRACKING_PASSWORD=<your-dagshub-access-token>

# Linux/macOS
export MLFLOW_TRACKING_URI="https://dagshub.com/Harsh05/MLOps-Lec1.mlflow"
export MLFLOW_TRACKING_USERNAME="<your-dagshub-username>"
export MLFLOW_TRACKING_PASSWORD="<your-dagshub-access-token>"
```

DagsHub DVC remote credentials are stored in `.dvc/config.local` (gitignored). Generate a token at DagsHub -> Settings -> Access Tokens.

> Windows note: set `PYTHONIOENCODING=utf-8` before running to avoid console encoding errors from MLflow output.

## Run the Pipeline

```bash
# Track raw data with DVC
dvc add data/raw/data.csv

# Run the full pipeline end-to-end
dvc repro

# Check pipeline status
dvc status

# Push data & models to DagsHub DVC storage
dvc push
```

### Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| pre_process | `python src/pre_process.py` | Reads raw CSV, writes `data/preprocessed/data.csv` |
| train | `python src/train.py` | GridSearchCV + train, saves `models/model.pkl`, logs to MLflow |
| evaluate | `python src/evaluate.py` | Loads model, computes accuracy, logs to MLflow |

## MLflow Experiment Tracking

Each `train` and `evaluate` run is logged to the DagsHub MLflow server:

- Metrics: `accuracy`, `eval_accuracy`
- Params: `best_n_estimators`, `best_max_depth`, `best_min_samples_split`
- Artifacts: `confusion_matrix.txt`, `classification_report.txt`, serialized `model`

View experiments at: `https://dagshub.com/Harsh05/MLOps-Lec1.mlflow`

## Results

Best model performance on the held-out test split (20%):

| Metric | Value |
|--------|-------|
| Accuracy | ~0.75 |

## License

This project is for educational purposes.
