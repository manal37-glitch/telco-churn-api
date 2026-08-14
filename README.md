# Telco Customer Churn Prediction API

A REST API that predicts whether a telecom customer will churn, built on the [IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) and served with FastAPI. Deployed on [Render](https://render.com).

## Problem

Customer churn is expensive to replace, so predicting which customers are at risk lets a telecom company intervene (retention offers, support outreach) before they leave. This project trains a binary classifier on customer account and service data (tenure, contract type, billing, add-on services) to output a churn probability.

## Pipeline

**1. Preprocessing**
- Dropped `customerID` (identifier, no predictive value)
- Coerced blank `TotalCharges` values to `0.0` and cast to float
- Collapsed `"No internet service"` / `"No phone service"` into `"No"` for the relevant service columns, so each is a clean binary feature
- Encoded binary Yes/No columns (and `gender`) as 0/1
- One-hot encoded `InternetService`, `Contract`, `PaymentMethod`
- Scaled `tenure`, `MonthlyCharges`, `TotalCharges` with `MinMaxScaler`

**2. Class imbalance**
The dataset is ~73% no-churn / 27% churn. Applied **SMOTE** to the training split only (after the train/test split, so the test set stays untouched and representative of real-world distribution).

**3. Model**
Logistic Regression (`solver='liblinear'`), trained on the SMOTE-balanced training set.

**4. Evaluation** (held-out 20% test set, 1,409 customers)

| Metric | Value |
|---|---|
| Accuracy | 0.774 |
| ROC AUC | 0.856 |
| Precision (churn) | 0.551 |
| Recall (churn) | 0.777 |
| F1 (churn) | 0.645 |

Recall on the churn class is prioritized over precision here — missing an at-risk customer (false negative) is more costly than flagging a customer who wasn't going to leave anyway (false positive), so the SMOTE rebalancing trades some precision for a lot more recall (0.777 vs. what a naive model on imbalanced data would give, typically <0.5).

## API

Built with FastAPI. Endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/predict` | Returns churn prediction + probabilities |

**Example request**

```bash
curl -X POST https://<your-render-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.5,
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
  }'
```

**Example response**

```json
{
  "churn_prediction": 1,
  "probability_no_churn": 0.34,
  "probability_yes_churn": 0.66
}
```

The `/predict` endpoint validates input with a Pydantic schema, applies the same preprocessing used in training (via the saved `scaler.joblib`), aligns columns to the exact order the model was trained on, and returns both the class prediction and class probabilities.

## Project structure

```
.
├── app.py                  # FastAPI app: preprocessing + inference
├── model_pipeline.joblib   # Trained LogisticRegression model
├── scaler.joblib           # Fitted MinMaxScaler (tenure, MonthlyCharges, TotalCharges)
├── requirements.txt
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Visit `http://localhost:8000/docs` for interactive Swagger docs.

## Deployment

Deployed on Render as a web service:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`


## Tech stack

Python, pandas, scikit-learn, imbalanced-learn (SMOTE), FastAPI, Pydantic, joblib, Render.
