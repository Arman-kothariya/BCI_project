# model_trainer.py

# This script trains multiple ML models, evaluates them, saves trained models,
# and also logs performance results with timestamps into a JSON file for future reference.

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # To save trained models
import os      # For directory handling
import json    # To store results in a .json file
from datetime import datetime  # For adding timestamps to results

def train_and_evaluate_models(X_train, X_test, y_train, y_test, config):
    """
    Trains SVM, Random Forest, and XGBoost models using provided training data,
    evaluates them on test data, saves the models and logs the results to a JSON file.
    
    Args:
        X_train, X_test, y_train, y_test: Training and testing data.
        config: Dictionary with hyperparameters and save path config.

    Returns:
        Dictionary containing accuracy and classification report for each model.
    """

    results = {}  # Store all model results

    # Define models using parameters from the config
    models = {
        'SVM': SVC(**config.get('svm_params', {})),
        'RandomForest': RandomForestClassifier(**config.get('rf_params', {})),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **config.get('xgb_params', {}))
    }

    # Path to save the trained models (default is 'saved_models' folder)
    model_save_path = config.get('model_save_path', 'saved_models')
    os.makedirs(model_save_path, exist_ok=True)

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate predictions
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Print performance
        print(f"{name} Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))  # Pretty text output

        # Save model details in results dictionary
        results[name] = {
            'model': name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp of this run
            'accuracy': accuracy,
            'report': report
        }

        # Save model to disk
        model_file = os.path.join(model_save_path, f"{name}_model.pkl")
        joblib.dump(model, model_file)
        print(f"{name} model saved to: {model_file}")

    # ---------------------- Save Results to JSON ----------------------

    # Directory and path to save results
    results_path = os.path.join('results', 'results.json')
    os.makedirs('results', exist_ok=True)

    # Load previous results if results.json already exists
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    # Generate a unique run ID using timestamp
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    existing_data[run_id] = results

    # Write updated results back to results.json
    with open(results_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"\n All model results saved to {results_path}")

    return results
