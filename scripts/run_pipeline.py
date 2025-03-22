import os
import subprocess
import time
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_pipeline():
    """
    Run the complete fraud detection pipeline:
    1. Data preprocessing
    2. Model training
    3. Start the FastAPI server
    """
    print("Starting fraud detection pipeline...")
    
    # Create necessary directories
    os.makedirs(os.path.join('..', 'data', 'raw'), exist_ok=True)
    os.makedirs(os.path.join('..', 'data', 'processed'), exist_ok=True)
    os.makedirs(os.path.join('..', 'models'), exist_ok=True)
    
    # Run data preprocessing
    print("\nRunning data preprocessing...")
    subprocess.run(['python', 'data_preprocessing.py'], cwd=os.path.join(os.path.dirname(os.path.abspath(__file__))))
    
    # Run model training
    print("\nTraining model...")
    subprocess.run(['python', 'model_training.py'], cwd=os.path.join(os.path.dirname(os.path.abspath(__file__))))
    
    # Start FastAPI server
    print("\nStarting FastAPI server...")
    # First, try to kill any existing uvicorn processes
    try:
        subprocess.run(['pkill', '-f', 'uvicorn'], check=False)
        time.sleep(1)  # Give it time to shut down
    except Exception as e:
        print(f"Warning: Could not kill existing uvicorn processes: {e}")
        
    # Now start the server on a specific port
    subprocess.Popen(
        ['uvicorn', 'app.main:app', '--reload', '--host', '127.0.0.1', '--port', '8000'], 
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    
    # Wait for server to start
    time.sleep(3)  # Give it more time to start
    
    # Run API tests
    print("\nRunning API tests...")
    subprocess.run(['python', 'test_api.py'], cwd=os.path.join(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    run_pipeline()
