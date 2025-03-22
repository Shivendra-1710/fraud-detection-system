import requests
import json
import os
import time
from requests.exceptions import RequestException

def test_api_endpoints():
    """
    Test the FastAPI endpoints
    """
    base_url = "http://localhost:8000"
    max_retries = 3
    retry_delay = 2
    
    # Test root endpoint
    print("\nTesting root endpoint...")
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            assert response.status_code == 200
            # Check if response contains HTML with the title of the fraud detection system
            assert "AI-Powered Fraud Detection System" in response.text
            print("Root endpoint test passed!")
            break
        except (RequestException, AssertionError) as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to connect to the API after multiple attempts.")
                return
    
    # Test single prediction
    print("\nTesting single prediction...")
    try:
        test_transaction = {
            "transaction_id": "TX123456789",
            "step": 1,
            "amount": 9839.64,
            "oldbalanceOrg": 170136.0,
            "newbalanceOrig": 160296.36,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "type": "PAYMENT",
            "isFlaggedFraud": 0
        }
        
        response = requests.post(f"{base_url}/predict", json=test_transaction, timeout=5)
        assert response.status_code == 200
        result = response.json()
        assert "fraud_probability" in result
        assert "is_fraud" in result
        assert "confidence_score" in result
        print("Single prediction test passed!")
    except (RequestException, AssertionError) as e:
        print(f"Single prediction test failed: {str(e)}")
    
    # Test batch prediction
    print("\nTesting batch prediction...")
    try:
        test_transactions = [
            test_transaction,
            {
                "transaction_id": "TX987654321",
                "step": 1,
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "type": "TRANSFER",
                "isFlaggedFraud": 0
            }
        ]
    
        response = requests.post(f"{base_url}/batch_predict", json=test_transactions, timeout=5)
        assert response.status_code == 200
        result = response.json()
        # Check that the response matches the BatchPredictionResponse schema
        assert "predictions" in result
        assert isinstance(result["predictions"], list)
        assert len(result["predictions"]) == 2
        # Check that each prediction has the expected fields
        for prediction in result["predictions"]:
            assert "fraud_probability" in prediction
            assert "is_fraud" in prediction
            assert "confidence_score" in prediction
        print("Batch prediction test passed!")
    except (RequestException, AssertionError) as e:
        print(f"Batch prediction test failed: {str(e)}")

def main():
    test_api_endpoints()

if __name__ == "__main__":
    main()
