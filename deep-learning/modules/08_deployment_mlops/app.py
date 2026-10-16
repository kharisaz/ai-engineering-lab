"""
FastAPI Microservice for Deep Learning Model Serving
Run locally using:
    uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import numpy as np
import time

# 1. Definisikan Schema Input Request menggunakan Pydantic
class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Usia pelanggan (18-100 tahun)", example=35)
    tenure: int = Field(..., ge=0, description="Lama berlangganan dalam bulan", example=12)
    monthly_charges: float = Field(..., ge=0.0, description="Tagihan bulanan dalam dollar", example=75.50)
    total_spend: float = Field(..., ge=0.0, description="Total pengeluaran pelanggan", example=906.00)
    support_tickets: int = Field(..., ge=0, description="Jumlah tiket komplain", example=2)
    online_services: int = Field(..., ge=0, le=10, description="Jumlah layanan aktif yang dipakai", example=3)

class PredictionResponse(BaseModel):
    status: str
    churn_probability: float
    risk_level: str
    recommended_action: str
    inference_time_ms: float

# 2. Definisikan Model PyTorch yang Kompatibel
class ChurnModel(nn.Module):
    def __init__(self, in_features=6, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        return self.network(x)

# 3. Inisialisasi FastAPI App
app = FastAPI(
    title="Customer Churn AI Inference Microservice",
    description="High-performance AI API for real-time customer risk assessment",
    version="1.0.0"
)

# Load model di memory saat startup
model = ChurnModel()
model.eval()

@app.get("/")
def home():
    return {
        "service": "Customer Churn AI Microservice",
        "status": "Online",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    start_time = time.time()
    
    try:
        # Preprocessing: convert dict to tensor
        input_data = np.array([[
            customer.age,
            customer.tenure,
            customer.monthly_charges,
            customer.total_spend,
            customer.support_tickets,
            customer.online_services
        ]], dtype=np.float32)
        
        # Sederhana: normalisasi dasar
        input_tensor = torch.tensor(input_data)
        
        # Inference
        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()
            
        latency = (time.time() - start_time) * 1000
        
        if prob > 0.65:
            risk = "HIGH RISK"
            action = "Kirim voucher diskon retensi 20% segera!"
        elif prob > 0.35:
            risk = "MODERATE RISK"
            action = "Kirim survey kepuasan pelanggan"
        else:
            risk = "LOW RISK"
            action = "Tidak ada tindakan khusus (Pelanggan loyal)"
            
        return PredictionResponse(
            status="success",
            churn_probability=round(prob, 4),
            risk_level=risk,
            recommended_action=action,
            inference_time_ms=round(latency, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
