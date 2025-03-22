import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

def load_and_preprocess_data(use_sample=False, sample_size=200000):
    """
    Load and preprocess the online payment fraud detection dataset
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Load the dataset
    # Use absolute path to avoid issues with relative paths
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'fraudTest.csv')
    
    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please ensure the dataset is placed in the correct location.")
        print("You can download the dataset from Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1")
        print("Rename it to 'fraudTest.csv' and place it in the 'data/raw' directory.")
        # Create a small sample dataset for testing if the real one doesn't exist
        print("\nCreating a small synthetic dataset for testing purposes...")
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Create a synthetic dataset with similar structure
        synthetic_data = {
            'step': [1, 1, 1, 1, 1] * 1000,
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'] * 1000,
            'amount': np.random.uniform(10, 10000, 5000),
            'nameOrig': ['C' + str(i).zfill(10) for i in range(5000)],
            'oldbalanceOrg': np.random.uniform(0, 100000, 5000),
            'newbalanceOrig': np.random.uniform(0, 100000, 5000),
            'nameDest': ['C' + str(i).zfill(10) for i in range(5000, 10000)],
            'oldbalanceDest': np.random.uniform(0, 100000, 5000),
            'newbalanceDest': np.random.uniform(0, 100000, 5000),
            'isFraud': [0, 0, 1, 0, 0] * 1000,  # 20% fraud rate for testing
            'isFlaggedFraud': [0, 0, 0, 0, 0] * 1000
        }
        df = pd.DataFrame(synthetic_data)
        df.to_csv(data_path, index=False)
        print(f"Synthetic dataset created and saved to {data_path}")
    else:
        # Load the dataset if it exists
        df = pd.read_csv(data_path)
    
    # If use_sample is True, take a smaller sample for faster processing
    if use_sample and len(df) > sample_size:
        print(f"\nUsing a sample of {sample_size} records for faster processing")
        # Keep all fraud cases and sample from non-fraud cases
        fraud_df = df[df['isFraud'] == 1]
        non_fraud_df = df[df['isFraud'] == 0]
        
        # If we have more non-fraud cases than the sample size minus fraud cases, sample them
        if len(non_fraud_df) > (sample_size - len(fraud_df)):
            non_fraud_sample = non_fraud_df.sample(n=sample_size - len(fraud_df), random_state=42)
            df = pd.concat([fraud_df, non_fraud_sample])
        
        print(f"Sample contains {len(df)} records with {df['isFraud'].sum()} fraud cases")
    
    # Display basic information about the dataset
    print("\nDataset Overview:")
    print(f"Number of records: {len(df)}")
    print(f"Number of fraud cases: {df['isFraud'].sum()}")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")
    
    # Feature engineering
    # Add time-based features from the step column (represents time)
    df['hour'] = df['step'] % 24  # Assuming each step could represent an hour
    df['day'] = df['step'] // 24  # Derive day from steps
    
    # Select relevant features
    features = [
        'step',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'hour',
        'day',
        'isFlaggedFraud'
    ]
    
    # Select features and target
    X = df[features].fillna(0)
    y = df['isFraud']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use
    scaler_path = os.path.join('..', 'models', 'scaler.pkl')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def main():
    X_train_scaled, X_test_scaled, y_train, y_test, _ = load_and_preprocess_data()
    print("\nData preprocessing completed!")
    print(f"Training set size: {len(X_train_scaled)}")
    print(f"Test set size: {len(X_test_scaled)}")

if __name__ == "__main__":
    main()
