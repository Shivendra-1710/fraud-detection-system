#!/usr/bin/env python
"""
Generate feature statistics from the training data to use for anomaly detection.
This script analyzes the dataset and saves statistics about each feature to help
identify transactions that are significantly different from the training data.
"""

import os
import pickle
import pandas as pd
import numpy as np
import sys
import importlib.util

# Import data_preprocessing directly using the same method as model_training.py
spec = importlib.util.spec_from_file_location(
    "data_preprocessing", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_preprocessing.py")
)
data_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_preprocessing)
load_and_preprocess_data = data_preprocessing.load_and_preprocess_data

def generate_feature_statistics(data):
    """
    Generate statistics for each numeric feature in the dataset
    
    Args:
        data: DataFrame containing the training data
    
    Returns:
        Dictionary with feature statistics
    """
    feature_stats = {}
    
    # Features to analyze
    numeric_features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest'
    ]
    
    for feature in numeric_features:
        if feature in data.columns:
            # Calculate statistics
            values = data[feature].values
            feature_stats[feature] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'q25': float(np.percentile(values, 25)),
                'q50': float(np.percentile(values, 50)),
                'q75': float(np.percentile(values, 75)),
                'q99': float(np.percentile(values, 99))
            }
            
    return feature_stats

def main():
    print("Loading and preprocessing data...")
    
    # Load and preprocess data using the imported function
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data(use_sample=False)
    
    # Create a DataFrame from the scaled data for analysis
    columns = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'hour', 'day', 'isFlaggedFraud'
    ]
    
    # Convert scaled data back to original scale for more interpretable statistics
    if hasattr(scaler, 'inverse_transform'):
        try:
            print("Converting scaled data back to original scale...")
            X_train_unscaled = scaler.inverse_transform(X_train_scaled)
            train_data = pd.DataFrame(X_train_unscaled, columns=columns)
        except Exception as e:
            print(f"Could not inverse transform: {e}")
            train_data = pd.DataFrame(X_train_scaled, columns=columns)
    else:
        train_data = pd.DataFrame(X_train_scaled, columns=columns)
    
    # Add fraud label
    train_data['isFraud'] = y_train
    
    print("Generating feature statistics...")
    feature_stats = generate_feature_statistics(train_data)
    
    # Print summary of statistics
    print("\nFeature Statistics Summary:")
    for feature, stats in feature_stats.items():
        print(f"\n{feature}:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {stat_value}")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save statistics to file
    stats_path = os.path.join(models_dir, 'feature_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(feature_stats, f)
    
    print(f"\nFeature statistics saved to {stats_path}")

if __name__ == "__main__":
    main()
