# =========================
# EEG Emotion Classification Pipeline
# =========================

from modules.config_loader import load_config
from modules.data_loader import DatasetLoader
from modules.preprocessing import bandpass_filter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

import numpy as np
from scipy.stats import skew, kurtosis
from modules import model_trainer  # (Optional - for modular training if needed)

import os
import pandas as pd
import joblib

# =========================
# Feature Extraction Functions
# =========================

def extract_extended_stats_features(eeg_data):
    """
    Extracts statistical features from EEG data for each sample.
    Useful for tabular data formats (e.g., precomputed features).
    """
    means = np.mean(eeg_data, axis=1)
    variances = np.var(eeg_data, axis=1)
    skews = skew(eeg_data, axis=1)
    kurtoses = kurtosis(eeg_data, axis=1)
    medians = np.median(eeg_data, axis=1)
    ranges = np.ptp(eeg_data, axis=1)
    stds = np.std(eeg_data, axis=1)

    features = np.vstack([means, variances, skews, kurtoses, medians, ranges, stds]).T
    return features

def extract_raw_features(eeg_data):
    """
    Placeholder for raw EEG time-series feature extraction.
    Modify this function if you plan to extract time-domain or frequency-domain features.
    """
    return eeg_data


# =========================
# Main Pipeline Execution
# =========================

if __name__ == "__main__":
    # --------- Load Configuration ---------
    config_path = "config.yaml"
    config = load_config(config_path)

    dataset_key = config['use_dataset']  # This selects which dataset to load
    dataset_config = config['datasets'][dataset_key]

    test_size = dataset_config['test_size']
    random_state = dataset_config['random_state']
    data_format = dataset_config.get('format', 'tabular')  # Either 'tabular' or 'raw'

    # --------- Load EEG Data ---------
    loader = DatasetLoader(config)
    eeg_data, labels = loader.load_data()

    # --------- Preprocessing: Bandpass Filter (for raw time-series only) ---------
    bp_params = dataset_config.get('preprocessing', {}).get('bandpass', {})
    lowcut = bp_params.get('lowcut', 1)
    highcut = bp_params.get('highcut', 50)
    order = bp_params.get('order', 5)
    fs = dataset_config.get('sampling_rate', 128)

    if data_format == 'raw':
        eeg_data_filtered = bandpass_filter(eeg_data, lowcut, highcut, fs, order)
        print("Preprocessing done. Sample filtered data shape:", eeg_data_filtered.shape)
    else:
        eeg_data_filtered = eeg_data
        print("Loaded tabular data shape:", eeg_data_filtered.shape)

    # --------- Feature Extraction ---------
    if data_format == 'tabular':
        features = extract_extended_stats_features(eeg_data_filtered)
    elif data_format == 'raw':
        features = extract_raw_features(eeg_data_filtered)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    print("Extracted features shape:", features.shape)

    # --------- Train-Test Split ---------
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=random_state
    )

    # --------- Feature Scaling ---------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # =========================
    # Model 1: Tuned SVM
    # =========================
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(SVC(), svm_param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    svm_grid.fit(X_train_scaled, y_train)

    print("Best SVM parameters:", svm_grid.best_params_)
    y_pred_svm = svm_grid.predict(X_test_scaled)
    print("Tuned SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
    print("Tuned SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

    pd.DataFrame(classification_report(y_test, y_pred_svm, output_dict=True)).T.to_csv("results/svm_report.csv")
    joblib.dump(svm_grid.best_estimator_, "models/svm_model.pkl")

    # =========================
    # Model 2: Tuned Random Forest
    # =========================
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        rf_param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    rf_grid.fit(X_train_scaled, y_train)

    print("Best Random Forest parameters:", rf_grid.best_params_)
    y_pred_rf = rf_grid.predict(X_test_scaled)
    print("Tuned Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
    print("Tuned Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

    pd.DataFrame(classification_report(y_test, y_pred_rf, output_dict=True)).T.to_csv("results/rf_report.csv")
    joblib.dump(rf_grid.best_estimator_, "models/rf_model.pkl")

    # =========================
    # Model 3: XGBoost (30% Subsample for Speed)
    # =========================
    X_train_sub, y_train_sub = resample(
        X_train_scaled, y_train,
        replace=False,
        n_samples=int(0.3 * len(y_train)),
        random_state=random_state
    )

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_sub)
    y_test_encoded = le.transform(y_test)

    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
    xgb_clf.fit(X_train_sub, y_train_encoded)

    y_pred_xgb_encoded = xgb_clf.predict(X_test_scaled)
    y_pred_xgb = le.inverse_transform(y_pred_xgb_encoded)

    print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
    print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

    pd.DataFrame(classification_report(y_test, y_pred_xgb, output_dict=True)).T.to_csv("results/xgb_report.csv")
    joblib.dump(xgb_clf, "models/xgb_model.pkl")


# ========================================================
# HOW TO SWITCH TO A DIFFERENT DATASET (VERY IMPORTANT)
# ========================================================
# Step 1: Open config.yaml
# Step 2: Under "use_dataset", set the key to your target dataset name.
# Example:
#   use_dataset: deap
#
# Step 3: In the "datasets" section, ensure your target dataset is defined like this:
#   deap:
#     path: data/deap/
#     format: raw
#     sampling_rate: 128
#     test_size: 0.2
#     random_state: 42
#     preprocessing:
#       bandpass:
#         lowcut: 4
#         highcut: 45
#         order: 5
#
# Step 4: Make sure the file/folder exists at the path specified (e.g., data/deap/)
# Step 5: Run run_pipeline.py again.
#
#  If the dataset is "tabular", set format: tabular
#  If the dataset is "raw EEG", set format: raw
