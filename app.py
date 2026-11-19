import joblib
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from model import LstmModel

# -------------------------------------------------------------
# 1. Load Preprocessors & Device Configuration
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_scaler = joblib.load("artifacts/seq_scaler.joblib")
context_transformer = joblib.load("artifacts/context_transformer.joblib")

SEQ_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration"
]

# -------------------------------------------------------------
# 2. Load Checkpoint & Instantiate LstmModel with Saved Config
# -------------------------------------------------------------
checkpoint = torch.load("artifacts/session_intent_lstm.pt", map_location=device, weights_only=True)

model = LstmModel(**checkpoint["hyperparameteres"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

app = FastAPI(title="Session Purchase Intent API")

# -------------------------------------------------------------
# 3. Pydantic Input & Output Schemas
# -------------------------------------------------------------
class SessionFeatures(BaseModel):
    Administrative: int = Field(..., example=2)
    Administrative_Duration: float = Field(..., example=45.0)
    Informational: int = Field(..., example=1)
    Informational_Duration: float = Field(..., example=12.5)
    ProductRelated: int = Field(..., example=18)
    ProductRelated_Duration: float = Field(..., example=640.0)
    BounceRates: float = Field(..., example=0.01)
    ExitRates: float = Field(..., example=0.025)
    PageValues: float = Field(..., example=14.2)
    SpecialDay: float = Field(..., example=0.0)
    Month: str = Field(..., example="Nov")
    OperatingSystems: int = Field(..., example=2)
    Browser: int = Field(..., example=2)
    Region: int = Field(..., example=1)
    TrafficType: int = Field(..., example=2)
    VisitorType: str = Field(..., example="Returning_Visitor")
    Weekend: bool = Field(..., example=False)

class PredictionResponse(BaseModel):
    purchase_probability: float
    predicted_intent: str

# -------------------------------------------------------------
# 4. Prediction Endpoint
# -------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict_purchase_intent(session: SessionFeatures):
    df_input = pd.DataFrame([session.model_dump()])

    seq_scaled = seq_scaler.transform(df_input[SEQ_COLS]).reshape(1, 3, 2)
    ctx_encoded = context_transformer.transform(df_input)

    seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, device=device)
    ctx_tensor = torch.tensor(ctx_encoded, dtype=torch.float32, device=device)

    with torch.no_grad():
        prob = torch.sigmoid(model(seq_tensor, ctx_tensor)).item()

    return PredictionResponse(
        purchase_probability=round(prob, 4),
        predicted_intent="Purchase" if prob >= 0.5 else "Drop-off"
    )