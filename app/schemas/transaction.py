from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum

class TransactionType(str, Enum):
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    CASH_IN = "CASH_IN"

class Transaction(BaseModel):
    transaction_id: Optional[str] = Field(
        default=None, 
        description="Unique identifier for the transaction",
        example="TX123456789"
    )
    step: int = Field(
        ..., 
        description="Time step in hours (1 step = 1 hour)",
        example=1,
        ge=0
    )
    amount: float = Field(
        ..., 
        description="Transaction amount",
        example=9839.64,
        gt=0
    )
    oldbalanceOrg: float = Field(
        ..., 
        description="Original balance before transaction",
        example=170136.0,
        ge=0
    )
    newbalanceOrig: float = Field(
        ..., 
        description="New balance after transaction",
        example=160296.36,
        ge=0
    )
    oldbalanceDest: float = Field(
        ..., 
        description="Recipient old balance before transaction",
        example=0.0,
        ge=0
    )
    newbalanceDest: float = Field(
        ..., 
        description="Recipient new balance after transaction",
        example=0.0,
        ge=0
    )
    type: TransactionType = Field(
        ..., 
        description="Type of transaction",
        example=TransactionType.PAYMENT
    )
    isFlaggedFraud: Optional[int] = Field(
        default=0, 
        description="System flag for fraud (0 = not flagged, 1 = flagged)",
        example=0,
        ge=0,
        le=1
    )

class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Probability of fraud between 0 and 1")
    is_fraud: bool = Field(..., description="True if the transaction is classified as fraudulent")
    confidence_score: float = Field(..., description="Confidence in the prediction, between 0 and 1")
    is_anomalous: bool = Field(False, description="True if the transaction is anomalous compared to training data")
    anomaly_reason: str = Field("", description="Reason why the transaction is considered anomalous, if applicable")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
