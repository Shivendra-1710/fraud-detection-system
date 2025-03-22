import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import importlib.util

# Import data_preprocessing directly
spec = importlib.util.spec_from_file_location(
    "data_preprocessing", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_preprocessing.py")
)
data_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_preprocessing)
load_and_preprocess_data = data_preprocessing.load_and_preprocess_data

def train_model():
    """
    Train the fraud detection model
    Returns:
        tuple: (model, X_test_scaled, y_test)
    """
    # Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, _ = load_and_preprocess_data()
    
    print("Creating a reduced complexity model for faster training...")
    # Initialize Random Forest Classifier with reduced parameters for faster training
    rf = RandomForestClassifier(
        n_estimators=50,  # Reduced from 200
        max_depth=10,     # Reduced from 15
        min_samples_split=20, 
        min_samples_leaf=5, 
        random_state=42,
        class_weight='balanced_subsample',
        verbose=1,        # Add verbosity to show progress
        n_jobs=-1         # Use all available cores
    )
    
    # Train the model on the full dataset
    print(f"Training on full dataset with {len(X_train_scaled)} samples containing {sum(y_train)} fraud cases")
    print("Training model...")
    rf.fit(X_train_scaled, y_train)
    
    # Save the model
    model_path = os.path.join('..', 'models', 'fraud_detection_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    
    return rf, X_test_scaled, y_test

def evaluate_model(model, X_test_scaled, y_test):
    """
    Evaluate the trained model
    """
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    
    # Print AUC-ROC score
    print('\nAUC-ROC Score:', roc_auc_score(y_test, y_pred_proba))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Not Fraud', 'Fraud'],
               yticklabels=['Not Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('..', 'models', 'confusion_matrix.png'))
    plt.close()

def main():
    # Train and evaluate the model
    model, X_test_scaled, y_test = train_model()
    evaluate_model(model, X_test_scaled, y_test)
    print("\nModel training and evaluation completed!")
    print(f"Model saved to: {os.path.join('..', 'models', 'fraud_detection_model.pkl')}")

if __name__ == "__main__":
    main()
