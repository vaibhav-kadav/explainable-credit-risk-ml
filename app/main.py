from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np

app = FastAPI(title = 'credit risk score')
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR/"models"/"gb_calibrated_model.pkl") #gb_calibrated_model.pkl

FEATURES = [
    "limit_bal", "sex", "education", "marriage", "age",
    "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
    "credit_utilization", "avg_payment_delay", "payment_to_bill_ratio"
] 

class creditRequest(BaseModel):
    data: dict

@app.post("/predict")
def predict(req: creditRequest):
    try:
        x = np.array([[req.data[f] for f in FEATURES]])
    except KeyError as e:
        return {"error": f"Missing feature {e}"}
    
    prob = model.predict_proba(x)[0, 1]

    risk_label = "HIGH_RISK" if prob >= 0.5 else "LOW_RISK"

    return {
        "default_probability": round(float(prob), 4),
        "risk_label": risk_label
    }
    