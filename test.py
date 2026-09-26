import joblib
import pandas as pd
model = joblib.load('loan_approval_model.pkl')
sample = pd.DataFrame({
    "age": [35],
    "income_k": [75],
    "credit_score": [720]
})
print(model.predict(sample))