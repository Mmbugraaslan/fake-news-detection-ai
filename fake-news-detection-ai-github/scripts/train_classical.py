from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARTIFACT_DIR = ROOT_DIR / "data" / "artifacts" / "classical"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv")


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    model = build_pipeline()
    model.fit(train_df["text"], train_df["label"])

    val_predictions = model.predict(val_df["text"])
    test_predictions = model.predict(test_df["text"])

    metrics = {
        "model_type": "tfidf_logistic_regression",
        "validation_accuracy": float(accuracy_score(val_df["label"], val_predictions)),
        "validation_f1": float(f1_score(val_df["label"], val_predictions)),
        "test_accuracy": float(accuracy_score(test_df["label"], test_predictions)),
        "test_f1": float(f1_score(test_df["label"], test_predictions)),
        "test_report": classification_report(test_df["label"], test_predictions, output_dict=True),
    }

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()