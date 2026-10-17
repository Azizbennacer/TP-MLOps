"""Train and track a reproducible Breast Cancer Wisconsin classifier."""

import os
import platform
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the experiment configuration from YAML."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in config file: {config_path}")
    return config


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the built-in Breast Cancer Wisconsin dataset."""
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


def validate_dataset(features: pd.DataFrame, labels: pd.Series) -> None:
    """Check the feature and target data before model training."""
    if features.empty:
        raise ValueError("Le jeu de données ne contient aucune observation.")

    if len(features) != len(labels):
        raise ValueError("Le nombre d'observations et de cibles diffère.")

    if features.isnull().any().any() or labels.isnull().any():
        raise ValueError("Des valeurs manquantes ont été détectées.")

    if labels.nunique() < 2:
        raise ValueError("La cible doit contenir au moins deux classes.")

    if not all(dtype.kind in "if" for dtype in features.dtypes):
        raise TypeError("Toutes les variables doivent être numériques.")


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build preprocessing and classification as a single leakage-safe pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config["model"]["C"],
                    max_iter=config["model"]["max_iter"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )


def split_data(
    features: pd.DataFrame, labels: pd.Series, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into stratified training and test sets."""
    return train_test_split(
        features,
        labels,
        test_size=config["data"]["test_size"],
        random_state=config["seed"],
        stratify=labels,
    )


def train_and_evaluate(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict[str, Any],
) -> tuple[Pipeline, dict[str, float]]:
    """Fit the configured pipeline and calculate classification metrics."""
    pipeline = build_pipeline(config)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    return pipeline, metrics


def run_training(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, float]:
    """Run training, evaluation, and MLflow logging from a YAML configuration."""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    config = load_config(config_file)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    features, labels = load_dataset()
    validate_dataset(features, labels)
    x_train, x_test, y_train, y_test = split_data(features, labels, config)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "seed": config["seed"],
                "test_size": config["data"]["test_size"],
                "C": config["model"]["C"],
                "max_iter": config["model"]["max_iter"],
                "dataset": "Breast Cancer Wisconsin (scikit-learn)",
            }
        )
        mlflow.log_params(
            {
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn_version,
                "pandas_version": version("pandas"),
                "numpy_version": version("numpy"),
                "mlflow_version": version("mlflow"),
            }
        )

        pipeline, metrics = train_and_evaluate(
            x_train, x_test, y_train, y_test, config
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=["numpy.dtype"],
        )
        mlflow.log_artifact(str(config_file), artifact_path="config")

        if shutil.which("git"):
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=False,
                text=True,
            )
            if git_commit.returncode == 0 and git_commit.stdout.strip():
                mlflow.set_tag("git_commit", git_commit.stdout.strip())

    return metrics


if __name__ == "__main__":
    print("Métriques :", run_training())
