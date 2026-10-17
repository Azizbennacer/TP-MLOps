from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.datasets import load_breast_cancer

from src.train import (
    build_pipeline,
    load_config,
    load_dataset,
    run_training,
    split_data,
    validate_dataset,
)


def test_dataset_is_valid() -> None:
    features, labels = load_dataset()

    validate_dataset(features, labels)


def test_pipeline_can_fit_and_predict() -> None:
    config = load_config()
    dataset = load_breast_cancer(as_frame=True)
    x_train, x_test, y_train, _ = split_data(dataset.data, dataset.target, config)

    pipeline = build_pipeline(config)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    assert len(predictions) == len(x_test)
    assert set(predictions).issubset(set(dataset.target.unique()))


def test_validation_rejects_missing_feature_values() -> None:
    features, labels = load_dataset()
    invalid_features = features.copy()
    invalid_features.iloc[0, 0] = float("nan")

    with pytest.raises(ValueError, match="valeurs manquantes"):
        validate_dataset(invalid_features, labels)


def test_training_logs_metrics_and_model(tmp_path: Path) -> None:
    config = load_config()
    tracking_path = tmp_path / "mlruns"
    config["mlflow"]["tracking_uri"] = tracking_path.as_uri()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    metrics = run_training(config_path)

    assert set(metrics) == {"accuracy", "f1", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())

    import mlflow

    runs = mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment_name"]]
    )
    assert len(runs) == 1
    assert runs.loc[0, "metrics.accuracy"] == pytest.approx(metrics["accuracy"])
    assert list(tracking_path.rglob("MLmodel"))
