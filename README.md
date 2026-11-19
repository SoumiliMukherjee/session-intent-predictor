# Online Shopper Purchase Intention Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://session-intent-predictor-nkzdfidkhgtsyjtepuekue.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg)](https://fastapi.tiangolo.com/)

An end-to-end deep learning system that predicts e-commerce session conversion intent using a **Dual-Stream PyTorch Architecture (LSTM + ANN)**, tuned via **Optuna** with zero data leakage and deployed live on **Streamlit Community Cloud** alongside a **FastAPI** REST inference service.

### Live App
Try the interactive predictor directly in your browser:  
**[https://session-intent-predictor-nkzdfidkhgtsyjtepuekue.streamlit.app/](https://session-intent-predictor-nkzdfidkhgtsyjtepuekue.streamlit.app/)**

---

## Architecture & Engineering Highlights

- **Dual-Stream Deep Learning (`LstmModel`):**
  - **Temporal Funnel Stream (`nn.LSTM`):** Models sequential user progression across page categories (`Administrative` $\rightarrow$ `Informational` $\rightarrow$ `ProductRelated` page counts and durations).
  - **Static Context Stream (`nn.Sequential` ANN):** Encodes session engagement metrics (`BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`) and one-hot encoded visitor attributes (`Month`, `VisitorType`, `TrafficType`, `Weekend`, `Region`, `Browser`, `OperatingSystems`) using `Linear` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout`.
  - **Feature Fusion & Imbalance Handling:** Concatenates temporal and static representations into a dense classification head trained with class-weighted `BCEWithLogitsLoss` (`pos_weight` calibrated for the 84.5% drop-off vs. 15.5% purchase class imbalance).
- **Zero-Leakage Preprocessing Pipeline:** All `StandardScaler` and `ColumnTransformer` encoders are fitted strictly on training splits before transforming validation and held-out test sets.
- **Bayesian Hyperparameter Optimization:** Automated search across hidden dimensions, dropout, batch size, learning rate, and weight decay using **Optuna** with `MedianPruner`, achieving **0.9492 Validation ROC-AUC**.
- **Flexible Deployment Stack:**
  - **Cloud Web App (`frontend.py`):** Cached standalone inference (`@st.cache_resource`) running on Streamlit Community Cloud.
  - **REST Microservice (`app.py`):** Pydantic-validated `/predict` endpoint with interactive Swagger/OpenAPI docs (`/docs`).

---

## Project Structure

```text
session-intent-predictor/
├── artifacts/
│   ├── session_intent_lstm.pt         # Trained PyTorch model weights & hyperparameters
│   ├── seq_scaler.joblib              # Fitted StandardScaler for temporal funnel features
│   └── context_transformer.joblib     # Fitted ColumnTransformer for static session context
├── notebooks/
│   └── deep_learning_project.ipynb    # Leak-free preprocessing, Optuna tuning & evaluation
├── model.py                           # Shared PyTorch LstmModel architecture
├── app.py                             # FastAPI REST inference service
├── frontend.py                        # Streamlit interactive web dashboard
├── requirements.txt                   # Pinned Python dependencies
└── README.md
