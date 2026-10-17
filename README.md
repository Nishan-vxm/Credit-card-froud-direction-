# Credit-card-froud-direction-
Credit Card Fraud Detection using Artificial Neural Networks (ANN) with a Gradio-based interactive dashboard to predict potential fraudulent transactions and estimate fraud probability.


FraudGuard AI: Credit Card Fraud Detection
A Gradio dashboard that uses a trained Artificial Neural Network (ANN) and a fitted feature scaler to estimate whether a credit-card transaction may be fraudulent.
Important: This is an educational machine-learning project, not a banking-grade fraud detection service. Predictions are estimates and must not be treated as confirmed fraud decisions.
Features
Dark-friendly Gradio dashboard
30 transaction inputs (Time, V1–V28, Amount)
ANN inference using a saved Keras model
Feature scaling using the scaler fitted during training
Predicted fraud probability and configurable decision threshold
Clear warning for anonymized V1–V28 features
Project structure
credit-card-fraud-detection/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── credit_card_fraud_ann.keras   # local model artifact; not committed by default
└── credit_card_scaler.pkl        # local scaler artifact; not committed by default
Requirements
Python 3.10 or 3.11 is recommended
A trained binary ANN model saved as credit_card_fraud_ann.keras
A fitted scaler saved as credit_card_scaler.pkl
The model should return one sigmoid probability for the fraud class, and the scaler must have been fitted on the same 30 features in the exact order shown in app.py.
Run locally
Clone or download this repository.
Create and activate a virtual environment (recommended).
Install dependencies:
pip install -r requirements.txt
Put credit_card_fraud_ann.keras and credit_card_scaler.pkl in the project directory. These files are excluded by .gitignore by default.
Start the app:
python app.py
Open the local URL printed in the terminal.
Optional configuration
Set a different decision threshold or artifact paths using environment variables:
FRAUD_THRESHOLD=0.35 python app.py
You can also set MODEL_PATH and SCALER_PATH if the artifacts are stored elsewhere. Choose thresholds based on validation data and the relative cost of false positives and false negatives; do not tune them using the test set.
Input feature order
The app expects these 30 inputs in order:
Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount.
The target column Class is not an input. The common dataset convention is 0 = legitimate and 1 = fraud.
Model artifacts and GitHub
The trained model and scaler are intentionally ignored by Git because model files can be large and the dataset/model may have redistribution restrictions. Do not upload private data, API keys, credentials, or a dataset unless you have permission to redistribute it.
If you decide to publish the model artifact, first check its size and dataset terms. For large files, consider Git LFS or a model-hosting service, and update this README with the download instructions. Never commit the original transaction dataset unless its license and privacy conditions permit it.
Limitations
V1–V28 are anonymized features and generally cannot be meaningfully entered by hand.
The dashboard expects inputs in the same order and preprocessing format used during training.
A threshold of 0.50 is only a default; it may not be optimal for an imbalanced fraud dataset.
Accuracy alone can be misleading for imbalanced data. Evaluate precision, recall, F1-score, PR-AUC, and the confusion matrix.
The UI does not connect to a bank or process live card transactions.
Responsible use
Use this project for learning and demonstration. Do not use it as the sole basis for blocking payments, accusing a person of fraud, or making financial decisions.
License
Released under the MIT License. See LICENSE.
