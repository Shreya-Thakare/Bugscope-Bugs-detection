"""BugScope Prediction API."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

app = FastAPI(
    title="BugScope API",
    description="Software defect prediction service",
    version="1.0.0",
)

model = joblib.load(MODEL_DIR / "defect_model.joblib")
encoders = joblib.load(MODEL_DIR / "encoders.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
selector = joblib.load(MODEL_DIR / "selector.joblib")
feature_cols = joblib.load(MODEL_DIR / "feature_cols.joblib")
num_cols = joblib.load(MODEL_DIR / "num_cols.joblib")
cat_cols = joblib.load(MODEL_DIR / "cat_cols.joblib")


class ModuleMetrics(BaseModel):
    loc: float = Field(..., ge=1)
    loc_executable: float = 500
    loc_comment: float = 50
    loc_blank: float = 40
    num_lines: float = 600
    cyclomatic_complexity: float = Field(..., ge=1)
    essential_complexity: float = 5
    design_complexity: float = 4
    branch_count: float = 10
    decision_count: float = 8
    edge_count: float = 30
    node_count: float = 20
    num_operators: float = 1000
    num_operands: float = 800
    num_unique_operators: float = 20
    num_unique_operands: float = 50
    halstead_volume: float = 4000
    halstead_effort: float = 6000
    halstead_difficulty: float = 10
    call_pairs: float = 6
    condition_count: float = 8
    parameter_count: float = 2
    path_count: float = 30
    percent_comments: float = 8
    avg_block_length: float = 10
    code_churn: float = 10
    review_coverage: float = Field(0.7, ge=0, le=1)
    test_coverage: float = Field(0.65, ge=0, le=1)
    team_experience: float = Field(0.6, ge=0, le=1)
    language: str = "Java"
    module_type: str = "Core"


class PredictionOut(BaseModel):
    defective: bool
    defect_probability: float
    risk_band: str
    recommendation: str


def transform(payload: dict):
    df = pd.DataFrame([payload])
    for c in cat_cols:
        le = encoders[c]
        val = str(df[c].iloc[0])
        if val not in set(le.classes_):
            val = le.classes_[0]
        df[c] = le.transform([val])
    df[num_cols] = scaler.transform(df[num_cols])
    X = df[feature_cols].values
    return selector.transform(X)


@app.get("/")
def root():
    return {"service": "BugScope API", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOut)
def predict(body: ModuleMetrics):
    try:
        X = transform(body.dict())
        pred = bool(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        if proba >= 0.6:
            band, rec = "High", "Prioritize review, tests, and refactoring"
        elif proba >= 0.35:
            band, rec = "Medium", "Increase review coverage and targeted tests"
        else:
            band, rec = "Low", "Maintain current quality practices"
        return PredictionOut(
            defective=pred,
            defect_probability=round(proba, 4),
            risk_band=band,
            recommendation=rec,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
