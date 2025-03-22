from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional
import sys
import os
import json
from pathlib import Path

# Add the parent directory to the path to make imports work from any directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import our modules
from app.schemas.transaction import Transaction, PredictionResponse, BatchPredictionResponse, TransactionType
from app.services.prediction import FraudDetectionService

app = FastAPI(
    title="AI-Powered Fraud Detection System",
    description="An AI-based system for detecting fraudulent transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create templates directory if it doesn't exist
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

# Create static directory if it doesn't exist
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize the fraud detection service
fraud_service = FraudDetectionService()

# Create example transactions for quick testing
EXAMPLE_TRANSACTIONS = {
    "legitimate_payment": Transaction(
        transaction_id="TX123456789",
        step=1,
        amount=9839.64,
        oldbalanceOrg=170136.0,
        newbalanceOrig=160296.36,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
        type=TransactionType.PAYMENT,
        isFlaggedFraud=0
    ),
    "suspicious_transfer": Transaction(
        transaction_id="TX987654321",
        step=1,
        amount=181.0,
        oldbalanceOrg=181.0,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
        type=TransactionType.TRANSFER,
        isFlaggedFraud=0
    ),
    "large_cashout": Transaction(
        transaction_id="TX567891234",
        step=10,
        amount=10000.0,
        oldbalanceOrg=10000.0,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=10000.0,
        type=TransactionType.CASH_OUT,
        isFlaggedFraud=0
    )
}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Create an HTML template for the home page
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI-Powered Fraud Detection System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .card {
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .btn {
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 10px 15px;
                text-decoration: none;
                border-radius: 4px;
                margin-right: 10px;
                margin-bottom: 10px;
            }
            .btn:hover {
                background-color: #2980b9;
            }
        </style>
    </head>
    <body>
        <h1>AI-Powered Fraud Detection System</h1>
        
        <div class="card">
            <h2>Quick Test Examples</h2>
            <p>Click on any of these buttons to test the fraud detection system with pre-filled examples:</p>
            <a href="/examples/legitimate_payment" class="btn">Test Legitimate Payment</a>
            <a href="/examples/suspicious_transfer" class="btn">Test Suspicious Transfer</a>
            <a href="/examples/large_cashout" class="btn">Test Large Cash-out</a>
        </div>
        
        <div class="card">
            <h2>API Documentation</h2>
            <p>Explore the API using the interactive documentation:</p>
            <a href="/docs" class="btn">Swagger UI</a>
            <a href="/redoc" class="btn">ReDoc</a>
        </div>
        
        <div class="card">
            <h2>Submit Custom Transaction</h2>
            <p>Use the form below to submit a custom transaction for fraud detection:</p>
            <a href="/form" class="btn">Open Transaction Form</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/examples/{example_name}")
async def get_example(example_name: str):
    """Get a pre-defined example transaction and show its prediction"""
    if example_name not in EXAMPLE_TRANSACTIONS:
        raise HTTPException(status_code=404, detail=f"Example '{example_name}' not found")
    
    # Get the example transaction
    transaction = EXAMPLE_TRANSACTIONS[example_name]
    
    # Make a prediction
    result = fraud_service.predict(transaction)
    
    # Create HTML response with the transaction details and prediction result
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Example Transaction: {example_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            .card {{
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .result {{
                font-weight: bold;
                font-size: 1.2em;
                margin-top: 10px;
            }}
            .fraud {{
                color: #e74c3c;
            }}
            .legitimate {{
                color: #27ae60;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .btn {{
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 10px 15px;
                text-decoration: none;
                border-radius: 4px;
                margin-top: 20px;
            }}
            .btn:hover {{
                background-color: #2980b9;
            }}
        </style>
    </head>
    <body>
        <h1>Example Transaction: {example_name.replace('_', ' ').title()}</h1>
        
        <div class="card">
            <h2>Transaction Details</h2>
            <table>
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Transaction ID</td>
                    <td>{transaction.transaction_id}</td>
                </tr>
                <tr>
                    <td>Step</td>
                    <td>{transaction.step}</td>
                </tr>
                <tr>
                    <td>Amount</td>
                    <td>${transaction.amount:.2f}</td>
                </tr>
                <tr>
                    <td>Original Balance</td>
                    <td>${transaction.oldbalanceOrg:.2f}</td>
                </tr>
                <tr>
                    <td>New Original Balance</td>
                    <td>${transaction.newbalanceOrig:.2f}</td>
                </tr>
                <tr>
                    <td>Destination Old Balance</td>
                    <td>${transaction.oldbalanceDest:.2f}</td>
                </tr>
                <tr>
                    <td>Destination New Balance</td>
                    <td>${transaction.newbalanceDest:.2f}</td>
                </tr>
                <tr>
                    <td>Transaction Type</td>
                    <td>{transaction.type}</td>
                </tr>
                <tr>
                    <td>System Flagged</td>
                    <td>{'Yes' if transaction.isFlaggedFraud == 1 else 'No'}</td>
                </tr>
            </table>
        </div>
        
        <div class="card">
            <h2>Prediction Result</h2>
            <div class="result {{'fraud' if result['is_fraud'] else 'legitimate'}}">
                This transaction is {{'likely fraudulent' if result['is_fraud'] else 'likely legitimate'}}.
            </div>
            <p>Fraud Probability: {result['fraud_probability']:.4f}</p>
            <p>Confidence Score: {result['confidence_score']:.4f}</p>
        </div>
        
        <a href="/" class="btn">Back to Home</a>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/form", response_class=HTMLResponse)
async def transaction_form(request: Request):
    """Display a form for submitting a custom transaction"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Submit Transaction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input, select {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #3498db;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .btn {
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 10px 15px;
                text-decoration: none;
                border-radius: 4px;
                margin-top: 20px;
            }
            .btn:hover {
                background-color: #2980b9;
            }
        </style>
    </head>
    <body>
        <h1>Submit a Transaction for Fraud Detection</h1>
        
        <form action="/submit-form" method="post">
            <div class="form-group">
                <label for="transaction_id">Transaction ID (optional):</label>
                <input type="text" id="transaction_id" name="transaction_id" placeholder="e.g., TX123456789">
            </div>
            
            <div class="form-group">
                <label for="step">Time Step (hours):</label>
                <input type="number" id="step" name="step" min="0" value="1" required>
            </div>
            
            <div class="form-group">
                <label for="amount">Amount ($):</label>
                <input type="number" id="amount" name="amount" min="0.01" step="0.01" value="100.00" required>
            </div>
            
            <div class="form-group">
                <label for="oldbalanceOrg">Original Balance ($):</label>
                <input type="number" id="oldbalanceOrg" name="oldbalanceOrg" min="0" step="0.01" value="1000.00" required>
            </div>
            
            <div class="form-group">
                <label for="newbalanceOrig">New Original Balance ($):</label>
                <input type="number" id="newbalanceOrig" name="newbalanceOrig" min="0" step="0.01" value="900.00" required>
            </div>
            
            <div class="form-group">
                <label for="oldbalanceDest">Destination Old Balance ($):</label>
                <input type="number" id="oldbalanceDest" name="oldbalanceDest" min="0" step="0.01" value="0.00" required>
            </div>
            
            <div class="form-group">
                <label for="newbalanceDest">Destination New Balance ($):</label>
                <input type="number" id="newbalanceDest" name="newbalanceDest" min="0" step="0.01" value="100.00" required>
            </div>
            
            <div class="form-group">
                <label for="type">Transaction Type:</label>
                <select id="type" name="type" required>
                    <option value="PAYMENT">PAYMENT</option>
                    <option value="TRANSFER">TRANSFER</option>
                    <option value="CASH_OUT">CASH_OUT</option>
                    <option value="DEBIT">DEBIT</option>
                    <option value="CASH_IN">CASH_IN</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="isFlaggedFraud">System Flagged:</label>
                <select id="isFlaggedFraud" name="isFlaggedFraud">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            
            <button type="submit">Check for Fraud</button>
        </form>
        
        <a href="/" class="btn">Back to Home</a>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/submit-form", response_class=HTMLResponse)
async def submit_form(
    transaction_id: Optional[str] = Form(None),
    step: int = Form(...),
    amount: float = Form(...),
    oldbalanceOrg: float = Form(...),
    newbalanceOrig: float = Form(...),
    oldbalanceDest: float = Form(...),
    newbalanceDest: float = Form(...),
    type: str = Form(...),
    isFlaggedFraud: int = Form(0)
):
    """Process the form submission and return the prediction result"""
    # Create a transaction object from the form data
    transaction = Transaction(
        transaction_id=transaction_id,
        step=step,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
        type=type,
        isFlaggedFraud=isFlaggedFraud
    )
    
    # Make a prediction
    result = fraud_service.predict(transaction)
    
    # Create HTML response with the prediction result
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            .card {{
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .result {{
                font-weight: bold;
                font-size: 1.2em;
                margin-top: 10px;
            }}
            .fraud {{
                color: #e74c3c;
            }}
            .legitimate {{
                color: #27ae60;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .btn {{
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 10px 15px;
                text-decoration: none;
                border-radius: 4px;
                margin-top: 20px;
                margin-right: 10px;
            }}
            .btn:hover {{
                background-color: #2980b9;
            }}
        </style>
    </head>
    <body>
        <h1>Fraud Detection Result</h1>
        
        <div class="card">
            <h2>Transaction Details</h2>
            <table>
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Transaction ID</td>
                    <td>{transaction.transaction_id or 'Not provided'}</td>
                </tr>
                <tr>
                    <td>Step</td>
                    <td>{transaction.step}</td>
                </tr>
                <tr>
                    <td>Amount</td>
                    <td>${transaction.amount:.2f}</td>
                </tr>
                <tr>
                    <td>Original Balance</td>
                    <td>${transaction.oldbalanceOrg:.2f}</td>
                </tr>
                <tr>
                    <td>New Original Balance</td>
                    <td>${transaction.newbalanceOrig:.2f}</td>
                </tr>
                <tr>
                    <td>Destination Old Balance</td>
                    <td>${transaction.oldbalanceDest:.2f}</td>
                </tr>
                <tr>
                    <td>Destination New Balance</td>
                    <td>${transaction.newbalanceDest:.2f}</td>
                </tr>
                <tr>
                    <td>Transaction Type</td>
                    <td>{transaction.type}</td>
                </tr>
                <tr>
                    <td>System Flagged</td>
                    <td>{'Yes' if transaction.isFlaggedFraud == 1 else 'No'}</td>
                </tr>
            </table>
        </div>
        
        <div class="card">
            <h2>Prediction Result</h2>
            <div class="result {{'fraud' if result['is_fraud'] else 'legitimate'}}">
                This transaction is {{'likely fraudulent' if result['is_fraud'] else 'likely legitimate'}}.
            </div>
            <p>Fraud Probability: {result['fraud_probability']:.4f}</p>
            <p>Confidence Score: {result['confidence_score']:.4f}</p>
        </div>
        
        <a href="/form" class="btn">Submit Another Transaction</a>
        <a href="/" class="btn">Back to Home</a>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """API endpoint for predicting fraud for a single transaction"""
    try:
        result = fraud_service.predict(transaction)
        return PredictionResponse(
            fraud_probability=result["fraud_probability"],
            is_fraud=result["is_fraud"],
            confidence_score=result["confidence_score"],
            is_anomalous=result.get("is_anomalous", False),
            anomaly_reason=result.get("anomaly_reason", "")
        )
    except Exception as e:
        print(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(transactions: List[Transaction]):
    """API endpoint for predicting fraud for multiple transactions"""
    try:
        results = fraud_service.batch_predict(transactions)
        # Convert the results to PredictionResponse objects
        prediction_responses = [
            PredictionResponse(
                fraud_probability=result["fraud_probability"],
                is_fraud=bool(result["is_fraud"]),  # Ensure this is a Python bool, not numpy.bool_
                confidence_score=result["confidence_score"],
                is_anomalous=result.get("is_anomalous", False),
                anomaly_reason=result.get("anomaly_reason", "")
            ) for result in results
        ]
        return BatchPredictionResponse(predictions=prediction_responses)
    except Exception as e:
        print(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
