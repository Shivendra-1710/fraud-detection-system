import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
import pickle

# Default feature statistics for anomaly detection based on typical financial transaction patterns
# These values are based on common patterns in financial transaction data
_feature_stats = {
    'amount': {
        'mean': 179553.0,
        'std': 603858.0,
        'min': 0.0,
        'max': 92445516.0,
        'q25': 13.55,
        'q50': 74.87,
        'q75': 208.0,
        'q99': 1111864.0
    },
    'oldbalanceOrg': {
        'mean': 835641.0,
        'std': 2888242.0,
        'min': 0.0,
        'max': 59585040.0,
        'q25': 0.0,
        'q50': 14208.0,
        'q75': 107360.0,
        'q99': 8940000.0
    },
    'newbalanceOrig': {
        'mean': 855113.0,
        'std': 2908293.0,
        'min': 0.0,
        'max': 49585040.0,
        'q25': 0.0,
        'q50': 0.0,
        'q75': 144587.0,
        'q99': 9253364.0
    },
    'oldbalanceDest': {
        'mean': 1100701.0,
        'std': 3399180.0,
        'min': 0.0,
        'max': 356015089.0,
        'q25': 0.0,
        'q50': 0.0,
        'q75': 324784.0,
        'q99': 14239246.0
    },
    'newbalanceDest': {
        'mean': 1224996.0,
        'std': 3564366.0,
        'min': 0.0,
        'max': 356179278.0,
        'q25': 0.0,
        'q50': 0.0,
        'q75': 417334.0,
        'q99': 14928629.0
    }
}

def is_anomalous(transaction: Dict) -> Tuple[bool, float, str]:
    """Check if a transaction is anomalous compared to the training data
    
    Args:
        transaction: Dictionary containing transaction data
    
    Returns:
        Tuple of (is_anomalous, anomaly_score, reason)
    """
    
    anomaly_score = 0.0
    reasons = []
    
    # Check numeric features for extreme values
    for feature in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        if feature in transaction and feature in _feature_stats:
            value = transaction[feature]
            stats = _feature_stats[feature]
            
            # Check if value is extremely high (above 99th percentile)
            if value > stats.get('q99', stats['max'] * 0.9):
                # Calculate how many times higher than the 99th percentile
                ratio = value / max(stats.get('q99', stats['max'] * 0.9), 1.0)
                if ratio > 1.5:  # If more than 50% higher than 99th percentile
                    anomaly_score += min(ratio * 2, 20.0)  # Cap at 20 to avoid extreme scores
                    reasons.append(f"{feature} ({value:,.2f}) is {ratio:.1f}x higher than typical values")
            
            # Check if value is extremely high (above max)
            if value > stats['max']:
                z_score = (value - stats['mean']) / max(stats['std'], 1.0)  # Avoid division by zero
                anomaly_score += min(abs(z_score), 30.0)  # Cap to avoid extreme scores
                reasons.append(f"{feature} ({value:,.2f}) exceeds maximum observed value of {stats['max']:,.2f}")
            
            # Check if value is negative (which shouldn't happen for financial data)
            if value < 0:
                anomaly_score += 10.0  # Stronger penalty for negative values
                reasons.append(f"{feature} ({value:,.2f}) is negative")
    
    # Check for logical inconsistencies
    if 'oldbalanceOrg' in transaction and 'newbalanceOrig' in transaction:
        old_balance = transaction['oldbalanceOrg']
        new_balance = transaction['newbalanceOrig']
        amount = transaction.get('amount', 0)
        
        # Check if balance change doesn't match transaction amount (with some tolerance)
        if abs((old_balance - new_balance) - amount) > 0.01 * max(amount, 1.0):
            anomaly_score += 3.0
            reasons.append("Balance change doesn't match transaction amount")
    
    # Consider a transaction anomalous if the score is above a threshold
    is_anomalous = anomaly_score > 5.0
    reason = "; ".join(reasons) if reasons else "No anomalies detected"
    
    return is_anomalous, anomaly_score, reason

def preprocess_transaction(transaction: Dict) -> List[float]:
    """
    Preprocess a single transaction for fraud detection
    
    Args:
        transaction: Dictionary containing transaction data
    
    Returns:
        List of processed features
    """
    # Extract features based on the model training schema
    # The model was trained with these exact features in this order
    
    # Calculate hour and day from step
    hour = transaction['step'] % 24
    day = transaction['step'] // 24
    
    # Create features in the same order as during training
    features = [
        transaction['step'],
        transaction['amount'],
        transaction['oldbalanceOrg'],
        transaction['newbalanceOrig'],
        transaction['oldbalanceDest'],
        transaction['newbalanceDest'],
        hour,
        day,
        transaction.get('isFlaggedFraud', 0)
    ]
    
    return features

def preprocess_transactions(transactions: List[Dict]) -> np.ndarray:
    """
    Preprocess multiple transactions for fraud detection
    
    Args:
        transactions: List of transaction dictionaries
    
    Returns:
        Numpy array of processed features
    """
    processed = []
    for transaction in transactions:
        processed.append(preprocess_transaction(transaction))
    return np.array(processed)
