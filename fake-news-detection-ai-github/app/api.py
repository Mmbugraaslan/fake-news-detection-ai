from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.services.predictor import predictor_service

# Uygulama baslangicinda modeli yukle
try:
    predictor_service.load()
except Exception:
    pass  # Yüklenemezse endpoint'ler hata dondurecek

app = FastAPI(
    title="Fake News Detection API",
    description="Sahte haber tespiti icin FastAPI servisi - Yerel DistilBERT + Groq LLaMA ensemble",
    version="0.2.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor_service.model_loaded else "error",
        model_loaded=predictor_service.model_loaded,
        model_type=predictor_service.model_type,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        result = predictor_service.predict(request.text)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return PredictResponse(**result)