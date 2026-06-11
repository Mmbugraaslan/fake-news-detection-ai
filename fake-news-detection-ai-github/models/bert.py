from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from app.config import DISTILBERT_ARTIFACTS_DIR, LABEL_TRUE


@dataclass
class DistilBertFakeNewsModel:
    model: DistilBertForSequenceClassification
    tokenizer: DistilBertTokenizerFast
    model_name: str = "distilbert"

    @classmethod
    def load(cls, artifact_dir: Path | None = None) -> "DistilBertFakeNewsModel":
        target_dir = artifact_dir or DISTILBERT_ARTIFACTS_DIR
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(target_dir))
        model = DistilBertForSequenceClassification.from_pretrained(str(target_dir))
        model.eval()
        return cls(model=model, tokenizer=tokenizer)

    def predict(self, text: str) -> dict[str, object]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            label = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[label].item())

        return {
            "label": label,
            "label_name": "true" if label == LABEL_TRUE else "fake",
            "score": confidence,
            "model_name": self.model_name,
        }


BertModelPlaceholder = DistilBertFakeNewsModel