from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

from drowsiness_detection.utils.config import BaselineBModelConfig


def build_geometric_classifier(
    config: BaselineBModelConfig,
    *,
    use_class_weights: bool,
    seed: int,
) -> Any:
    architecture = config.architecture.lower()
    class_weight = "balanced" if use_class_weights else None

    if architecture == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=config.random_forest_n_estimators,
            max_depth=config.random_forest_max_depth,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=1,
        )

    if architecture in {"logistic_regression", "logistic"}:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=config.logistic_max_iter,
                class_weight=class_weight,
                random_state=seed,
            ),
        )

    if architecture in {"svm", "rbf_svm"}:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        return make_pipeline(
            StandardScaler(),
            SVC(
                C=config.svm_c,
                kernel="rbf",
                probability=True,
                class_weight=class_weight,
                random_state=seed,
            ),
        )

    raise ValueError(f"Unsupported Baseline B geometric classifier: {config.architecture}")


def save_geometric_classifier(model: Any, path: str | Path, metadata: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)


def load_geometric_classifier(path: str | Path) -> tuple[Any, dict[str, Any]]:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Baseline B geometric model does not exist: {model_path}. "
            "Train it with `python scripts/train_baseline_b.py`."
        )
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], dict(payload.get("metadata", {}))
    return payload, {}
