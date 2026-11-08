#!/usr/bin/env python3
"""Runtime entry point with the PCA-aware PLS search-space correction."""

from __future__ import annotations

import sys
from typing import Any

import downstream_regression as pipeline


def suggest_configuration(
    trial,
    model_name: str,
    n_features: int,
    n_samples: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    reduction = trial.suggest_categorical("embedding_reduction", ["raw", "pca"])
    reduction_params: dict[str, Any] = {}
    if reduction == "pca":
        upper = max(2, min(128, n_features, n_samples - 1))
        choices = sorted(
            {value for value in [16, 32, 64, 128] if value <= upper} | {upper}
        )
        reduction_params["n_components"] = trial.suggest_categorical(
            "pca_n_components", choices
        )
    effective_features = int(reduction_params.get("n_components", n_features))

    if model_name == "ridge":
        model_params = {"alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True)}
    elif model_name == "elasticnet":
        model_params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 1e1, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    elif model_name == "pls":
        model_params = {
            "n_components": trial.suggest_int(
                "pls_n_components",
                1,
                min(32, effective_features, n_samples - 1),
            )
        }
    elif model_name == "extratrees":
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", 0.3, 0.6, 1.0]
            ),
        }
    elif model_name == "histgradientboosting":
        model_params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.2, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 100, 300, step=100),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-6, 1e1, log=True
            ),
        }
    elif model_name == "svr_rbf":
        model_params = {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    else:
        raise ValueError(model_name)
    return reduction, reduction_params, model_params


pipeline.suggest_configuration = suggest_configuration


if __name__ == "__main__":
    sys.exit(pipeline.main())
