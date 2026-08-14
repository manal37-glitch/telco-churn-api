import os
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "model_pipeline.joblib"
SCALER_PATH = "scaler.joblib"

EXPECTED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService_DSL",
    "InternetService_Fiber optic",
    "InternetService_No",
    "Contract_Month-to-month",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "OnlineSecurity",
    "MultipleLines",
]

SCALE_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = ["InternetService", "Contract", "PaymentMethod"]


class CustomerInput(BaseModel):
    gender: str = Field(..., description="Female or Male")
    SeniorCitizen: int = Field(..., description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0)
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    PaperlessBilling: str = Field(..., description="Yes or No")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: Any = Field(..., description="Numeric value or blank string")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaymentMethod: str = Field(..., description="Bank transfer, Credit card, Electronic check, or Mailed check")


app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")


def _normalize_yes_no(value: Any) -> Any:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    text = str(value).strip()
    text_lower = text.lower()
    if text_lower in {"yes", "y"}:
        return 1
    if text_lower in {"no", "n"}:
        return 0
    return text


def _normalize_gender(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).strip().lower()
    if text in {"female", "f"}:
        return 1
    if text in {"male", "m"}:
        return 0
    raise ValueError(f"Unsupported gender value: {value!r}")


def preprocess_customer(payload: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([payload])

    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0", None: "0.0"})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    for col in [
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines",
        "OnlineSecurity",
    ]:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    for col in YES_NO_COLS:
        df[col] = df[col].map(_normalize_yes_no)

    df["gender"] = df["gender"].map(_normalize_gender)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Missing {SCALER_PATH}. The trained scaler must be saved alongside the model to match the training preprocessing."
        )

    scaler = joblib.load(SCALER_PATH)
    df[SCALE_COLUMNS] = scaler.transform(df[SCALE_COLUMNS])

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, dtype=int)

    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = 0

    return df[EXPECTED_COLUMNS]


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(payload: CustomerInput):
    try:
        features = preprocess_customer(payload.model_dump())
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)

        return {
            "churn_prediction": int(prediction[0]),
            "probability_no_churn": float(probabilities[0][0]),
            "probability_yes_churn": float(probabilities[0][1]),
        }
    except Exception as exc:  # pragma: no cover - API error branch
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
