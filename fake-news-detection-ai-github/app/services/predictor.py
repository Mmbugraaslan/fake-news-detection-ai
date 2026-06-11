from __future__ import annotations

import os
from pathlib import Path

import joblib
import torch
from groq import Groq
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import CLASSICAL_ARTIFACTS_DIR, CLASSICAL_MODEL_FILENAME, DISTILBERT_ARTIFACTS_DIR


class PredictorService:
    def __init__(self) -> None:
        self.classical_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.groq_client = None
        self.model_type: str | None = None
        self.model_loaded = False

    def load(self) -> None:
        self._load_distilbert()
        self._load_classical()
        self._load_groq()

        if self.bert_model is not None and self.bert_tokenizer is not None:
            self.model_type = "distilbert"
            self.model_loaded = True
            return

        if self.classical_model is not None:
            self.model_type = "classical"
            self.model_loaded = True
            return

        raise FileNotFoundError("Yuklenebilir egitilmis model bulunamadi.")

    def _load_distilbert(self) -> None:
        config_path = DISTILBERT_ARTIFACTS_DIR / "config.json"
        if not DISTILBERT_ARTIFACTS_DIR.exists() or not config_path.exists():
            return

        self.bert_tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_ARTIFACTS_DIR))
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_ARTIFACTS_DIR))
        self.bert_model.eval()

    def _load_classical(self) -> None:
        model_path = CLASSICAL_ARTIFACTS_DIR / CLASSICAL_MODEL_FILENAME
        if not CLASSICAL_ARTIFACTS_DIR.exists() or not model_path.exists():
            return

        self.classical_model = joblib.load(model_path)

    def _load_groq(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            # Config dosyasindan oku
            config_path = Path(__file__).resolve().parents[2] / "groq_config.txt"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            api_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                            break
        if api_key:
            self.groq_client = Groq(api_key=api_key)

    def predict(self, text: str) -> dict[str, object]:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Metin bos olamaz.")

        if not self.model_loaded:
            raise RuntimeError("Model henuz yuklenmedi.")

        # 1. Yerel model tahmini (DistilBERT veya klasik)
        local_result = None
        if self.bert_model is not None and self.bert_tokenizer is not None:
            local_result = self._predict_distilbert(normalized_text)
        elif self.classical_model is not None:
            local_result = self._predict_classical(normalized_text, fallback_used=True)

        # 2. Groq LLaMA tahmini
        groq_result = None
        if self.groq_client is not None:
            try:
                groq_result = self._predict_groq(normalized_text)
            except Exception:
                pass  # Groq hata verirse sadece yerel model sonucunu dondur

        # 3. Sonuclari birlestir
        return self._merge_results(local_result, groq_result, normalized_text)

    def _predict_distilbert(self, text: str) -> dict[str, object]:
        encoded = self.bert_tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.bert_model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            label_id = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[label_id].item())

        return self._build_response(label_id=label_id, confidence=confidence, model="distilbert", fallback_used=False)

    def _predict_classical(self, text: str, fallback_used: bool) -> dict[str, object]:
        probabilities = self.classical_model.predict_proba([text])[0]
        label_id = int(probabilities.argmax())
        confidence = float(probabilities[label_id])
        return self._build_response(label_id=label_id, confidence=confidence, model="classical", fallback_used=fallback_used)

    def _predict_groq(self, text: str) -> dict[str, object]:
        prompt = (
            "As a fake news detection expert, analyze the following news article and determine if it is REAL or FAKE.\n\n"
            f"Article: {text}\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            '{"label": "real" or "fake", "confidence": 0.0 to 1.0, "explanation": "brief reason"}\n'
            "Do not include any other text."
        )

        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a fake news detection AI. Respond only with JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )

        content = response.choices[0].message.content.strip()
        # JSON'i parse et
        import json
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # JSON degilse, icerikten cikarim yap
            label = "real" if "real" in content.lower() else "fake"
            result = {"label": label, "confidence": 0.7, "explanation": content[:200]}

        label_id = 1 if result.get("label", "").lower() == "real" else 0
        confidence = float(result.get("confidence", 0.7))
        explanation = result.get("explanation", "")

        return {
            "label": "real" if label_id == 1 else "fake",
            "label_tr": "gercek" if label_id == 1 else "sahte",
            "confidence": confidence,
            "model": "groq_llama",
            "fallback_used": False,
            "explanation": explanation,
        }

    def _build_response(self, label_id: int, confidence: float, model: str, fallback_used: bool) -> dict[str, object]:
        label = "real" if label_id == 1 else "fake"
        label_tr = "gercek" if label_id == 1 else "sahte"
        return {
            "label": label,
            "label_tr": label_tr,
            "confidence": confidence,
            "model": model,
            "fallback_used": fallback_used,
        }

    def _merge_results(self, local_result: dict | None, groq_result: dict | None, text: str) -> dict[str, object]:
        if groq_result is None:
            # Groq yoksa sadece yerel model
            return local_result if local_result else {"error": "Tahmin yapilamadi"}

        if local_result is None:
            # Yerel model yoksa sadece Groq
            return groq_result

        # Ikisi de varsa — birlestir
        local_label = local_result["label"]
        groq_label = groq_result["label"]
        local_conf = local_result["confidence"]
        groq_conf = groq_result["confidence"]

        # Uyumluysa guven artar, uyumsuzsa dusuk guven
        if local_label == groq_label:
            final_label = local_label
            final_conf = min(0.99, max(local_conf, groq_conf) + 0.1)
            agreement = "both_agree"
        else:
            # Guveni yuksek olani tercih et
            if local_conf >= groq_conf:
                final_label = local_label
                final_conf = local_conf * 0.8  # Uyumsuzluk = dusuk guven
            else:
                final_label = groq_label
                final_conf = groq_conf * 0.8
            agreement = "disagree"

        return {
            "label": final_label,
            "label_tr": "gercek" if final_label == "real" else "sahte",
            "confidence": round(final_conf, 4),
            "local_model": {
                "label": local_label,
                "confidence": round(local_conf, 4),
                "model": local_result["model"],
            },
            "llm_analysis": {
                "label": groq_label,
                "confidence": round(groq_conf, 4),
                "explanation": groq_result.get("explanation", ""),
            },
            "agreement": agreement,
            "model": "ensemble",
            "fallback_used": False,
        }


predictor_service = PredictorService()
