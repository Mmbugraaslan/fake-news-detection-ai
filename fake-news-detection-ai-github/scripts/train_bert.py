from pathlib import Path
import json
import sys
import os
import io

# Arka plan çalıştırmasında stdout/stderr None olabilir — düzelt
if sys.stdout is None:
    sys.stdout = io.TextIOWrapper(open(os.devnull, "wb"), encoding="utf-8")
if sys.stderr is None:
    sys.stderr = io.TextIOWrapper(open(os.devnull, "wb"), encoding="utf-8")

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARTIFACT_DIR = ROOT_DIR / "data" / "artifacts" / "distilbert"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 2


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv")


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args_kwargs = dict(
        output_dir=str(ARTIFACT_DIR / "checkpoints"),

        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        use_cpu=True,
    )
    try:
        training_args = TrainingArguments(**training_args_kwargs)
    except TypeError:
        training_args_kwargs["evaluation_strategy"] = training_args_kwargs.pop("eval_strategy")
        training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    validation_metrics = trainer.evaluate(val_dataset)
    test_metrics = trainer.evaluate(test_dataset)

    trainer.save_model(str(ARTIFACT_DIR))
    tokenizer.save_pretrained(str(ARTIFACT_DIR))

    metrics = {
        "model_type": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "train_rows": int(len(train_df)),
        "validation": {key: float(value) for key, value in validation_metrics.items() if isinstance(value, (int, float))},
        "test": {key: float(value) for key, value in test_metrics.items() if isinstance(value, (int, float))},
    }

    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()