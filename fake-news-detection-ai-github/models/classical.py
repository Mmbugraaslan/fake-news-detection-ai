from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import CLASSICAL_ARTIFACTS_DIR, CLASSICAL_MODEL_FILENAME, LABEL_FAKE, LABEL_TRUE


@dataclass
class ClassicalFakeNewsModel:
    pipeline: Pipeline
    model_name: str = "classical"

    @classmethod
    def build(cls) -> "ClassicalFakeNewsModel":
        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        max_features=50000,
                        min_df=2,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )
        return cls(pipeline=pipeline)

    @classmethod
    def load(cls, artifact_dir: Path | None = None) -> "ClassicalFakeNewsModel":
        target_dir = artifact_dir or CLASSICAL_ARTIFACTS_DIR
        pipeline = joblib.load(target_dir / CLASSICAL_MODEL_FILENAME)
        return cls(pipeline=pipeline)

    def train(self, texts: list[str], labels: list[int]) -> None:
        self.pipeline.fit(texts, labels)

    def predict(self, text: str) -> dict[str, object]:
        probabilities = self.pipeline.predict_proba([text])[0]
        label = int(np.argmax(probabilities))
        confidence = float(probabilities[label])
        return {
            "label": label,
            "label_name": self._label_to_name(label),
            "score": confidence,
            "model_name": self.model_name,
        }

    def save(self, artifact_dir: Path | None = None) -> Path:
        target_dir = artifact_dir or CLASSICAL_ARTIFACTS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / CLASSICAL_MODEL_FILENAME
        joblib.dump(self.pipeline, output_path)
        return output_path

    @staticmethod
    def _label_to_name(label: int) -> str:
        return "true" if label == LABEL_TRUE else "fake"


ClassicalModelPlaceholder = ClassicalFakeNewsModel