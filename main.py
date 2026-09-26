from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(
    level=logging.INFO
)
prediction_count = 0
approved_count = 0
rejected_count = 0

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Loan Approval Prediction API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load(
    "loan_approval_model.pkl"
)


class LoanRequest(BaseModel):

    age: int
    income_k: float
    credit_score: int


@app.get("/")
def home():

    return {
        "message": "Loan Approval API is running"
    }
@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: LoanRequest):

    

    input_data = pd.DataFrame([{
        "age": data.age,
        "income_k": data.income_k,
        "credit_score": data.credit_score
    }])

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    logger.info(
        "Prediction | age=%s income=%s credit_score=%s prediction=%s probability=%.4f",
        data.age,
        data.income_k,
        data.credit_score,
        prediction,
        probability
    )

    global prediction_count
    global approved_count
    global rejected_count

    prediction_count += 1




    msg=''
    if prediction == 1:
        msg=' Approved'
        approved_count += 1
    else:
        msg='Not approved'
        rejected_count += 1

    return {
        'msg': msg,
        "approved": int(prediction),
        "approval_probability": float(probability)
    }

@app.get("/metrics")
def metrics():

    return {
        "total_predictions": prediction_count,
        "approved": approved_count,
        "rejected": rejected_count
    }