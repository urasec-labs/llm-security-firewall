"""Train and save a classifier from the bundled demonstration corpus.

Run from the workspace root with `PYTHONPATH=artifacts/api-server` set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from firewall.detectors.classifier import build_pipeline
from firewall.detectors.training_data import training_labels, training_texts


def train(output_path: Path) -> Path:
    """Fit the TF-IDF/logistic-regression baseline and save the pipeline."""
    pipeline = build_pipeline()
    pipeline.fit(training_texts(), training_labels())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/api-server/models/prompt-risk.joblib"),
    )
    arguments = parser.parse_args()
    saved_path = train(arguments.output)
    print(f"Saved demonstration classifier to {saved_path}")


if __name__ == "__main__":
    main()