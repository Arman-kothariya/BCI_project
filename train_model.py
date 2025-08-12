import pandas as pd
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

# Paths
DATA_PATH = "datasets/kaggle_temp/emotion.csv"
MODEL_PATH = "models/best_model.pkl"
RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
METRIC_JSON = os.path.join(RESULTS_DIR, "model_metrics.json")
METRIC_CSV = os.path.join(RESULTS_DIR, "model_metrics.csv")

os.makedirs(PLOT_DIR, exist_ok=True)

# Load EEG dataset
print("[INFO] Loading EEG dataset...")
df = pd.read_csv(DATA_PATH)

if 'label' not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

X = df.drop('label', axis=1)
y = df['label']

# Train/test split
print("[INFO] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Binarize labels for ROC if binary
classes = y.unique()
is_binary = len(classes) == 2
if is_binary:
    y_test_bin = label_binarize(y_test, classes=[min(classes), max(classes)])

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

metrics_dict = {}
conf_matrices = {}

# Train and evaluate
for name, clf in models.items():
    print(f"[INFO] Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics_dict[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

    # Confusion Matrix
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

    # ROC curve (only if binary)
    if is_binary:
        y_score = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title(f"ROC Curve - {name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{name}_roc_curve.png"))
        plt.close()

# Save best model
best_model_name = max(metrics_dict, key=lambda k: metrics_dict[k]["accuracy"])
best_model = models[best_model_name]
joblib.dump(best_model, MODEL_PATH)
print(f"[INFO] Best model '{best_model_name}' saved to {MODEL_PATH}")

# Save metrics to JSON & CSV
with open(METRIC_JSON, "w") as f:
    json.dump(metrics_dict, f, indent=4)

pd.DataFrame(metrics_dict).T.to_csv(METRIC_CSV)

# Plot Accuracy Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(metrics_dict.keys()), y=[m['accuracy'] for m in metrics_dict.values()])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "accuracy_comparison.png"))
plt.close()

# Plot F1, Precision, Recall Comparison
for metric_name in ['f1_score', 'precision', 'recall']:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(metrics_dict.keys()), y=[m[metric_name] for m in metrics_dict.values()])
    plt.title(f"Model {metric_name.replace('_', ' ').title()} Comparison")
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric_name}_comparison.png"))
    plt.close()

# Plot Confusion Matrix for each model
for name, cm in conf_matrices.items():
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_confusion_matrix.png"))
    plt.close()
