# AI-Powered Fraud Detection System

An advanced fraud detection system using machine learning and deep learning techniques to identify fraudulent activities in online transactions.

## Features

- Real-time transaction monitoring
- Advanced anomaly detection using machine learning
- Deep learning-based pattern recognition
- Risk scoring system
- Transaction categorization
- Interactive dashboard for monitoring

## Project Structure

```
├── app/              # FastAPI application
│   ├── main.py      # Main application file
│   ├── schemas/     # Data models
│   ├── services/    # Business logic
│   └── utils/       # Utility functions
├── scripts/         # Utility scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── run_pipeline.py
│   └── test_api.py
├── data/            # Dataset storage
│   ├── raw/        # Raw dataset
│   └── processed/  # Processed data
├── models/          # Trained models
└── requirements.txt # Project dependencies
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python scripts/run_pipeline.py
```

## Technologies Used

- Python
- TensorFlow
- Scikit-learn
- Flask
- Pandas
- NumPy

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
