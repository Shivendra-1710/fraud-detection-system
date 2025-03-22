import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.transaction import Transaction

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to AI-Powered Fraud Detection System"}

def test_predict_fraud():
    test_transaction = {
        "amount": 100.0,
        "time": 1000,
        "v1": 0.5,
        "v2": -0.2,
        "v3": 0.8,
        "v4": -0.5,
        "v5": 0.3,
        "v6": -0.1,
        "v7": 0.4,
        "v8": -0.3,
        "v9": 0.6,
        "v10": -0.4,
        "v11": 0.7,
        "v12": -0.2,
        "v13": 0.5,
        "v14": -0.1,
        "v15": 0.3,
        "v16": -0.2,
        "v17": 0.4,
        "v18": -0.3,
        "v19": 0.6,
        "v20": -0.4,
        "v21": 0.7,
        "v22": -0.2,
        "v23": 0.5,
        "v24": -0.1,
        "v25": 0.3,
        "v26": -0.2,
        "v27": 0.4,
        "v28": -0.3
    }

    response = client.post("/predict", json=test_transaction)
    assert response.status_code == 200
    result = response.json()
    assert "fraud_probability" in result
    assert "is_fraud" in result
    assert "confidence_score" in result

def test_batch_predict():
    test_transactions = [
        {
            "amount": 100.0,
            "time": 1000,
            "v1": 0.5,
            "v2": -0.2,
            # ... other fields
        },
        {
            "amount": 200.0,
            "time": 2000,
            "v1": 0.6,
            "v2": -0.3,
            # ... other fields
        }
    ]

    response = client.post("/batch_predict", json=test_transactions)
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 2
