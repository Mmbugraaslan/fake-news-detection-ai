from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Siniflandirilacak haber metni")


class LocalModelResult(BaseModel):
    label: str
    confidence: float
    model: str


class LLMAnalysisResult(BaseModel):
    label: str
    confidence: float
    explanation: str


class PredictResponse(BaseModel):
    label: str
    label_tr: str
    confidence: float
    local_model: LocalModelResult | None = None
    llm_analysis: LLMAnalysisResult | None = None
    agreement: str | None = None
    model: str
    fallback_used: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str | None
