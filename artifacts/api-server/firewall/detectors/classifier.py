"""Lightweight TF-IDF/logistic-regression classifier for prompt risk scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from firewall.detectors.training_data import training_labels, training_texts


def build_pipeline() -> Pipeline:
    """Create a small, interpretable baseline classifier."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=12_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=23,
                ),
            ),
        ]
    )


class PromptClassifier:
    """Load a trusted trained model or fit the included demo corpus once."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._pipeline: Pipeline | None = None

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def load_or_train(self) -> None:
        if self.model_path:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Configured classifier file does not exist: {self.model_path}"
                )
            # Only load joblib files produced by a trusted training pipeline.
            import joblib

            loaded: Any = joblib.load(self.model_path)
            if not isinstance(loaded, Pipeline):
                raise TypeError(
                    "The configured classifier file is not a sklearn Pipeline."
                )
            self._pipeline = loaded
            return

        pipeline = build_pipeline()
        pipeline.fit(training_texts(), training_labels())
        self._pipeline = pipeline

    def attack_score(self, text: str) -> float:
        if self._pipeline is None:
            raise RuntimeError("Classifier has not been initialized.")
        probabilities = self._pipeline.predict_proba([text])[0]
        classes = list(self._pipeline.classes_)
        return float(probabilities[classes.index(1)])