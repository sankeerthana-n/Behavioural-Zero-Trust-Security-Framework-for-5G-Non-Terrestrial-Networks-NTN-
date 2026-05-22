"""
Isolation Forest Model Trainer
==============================
Trains an Isolation Forest on normal NTN telemetry and evaluates its
ability to detect spoofing/replay/impersonation attacks.

Outputs:
  models/isolation_forest_model.pkl   - Trained model
  charts/confusion_matrix.png         - Confusion matrix heatmap
  charts/roc_curve.png                - ROC curve with AUC
  charts/precision_recall_curve.png   - Precision-Recall curve
  charts/feature_importance.png       - Feature importance bar chart
  charts/anomaly_score_distribution.png - Score distributions
  charts/classification_report.txt    - Text classification report

Usage:
  python train_model.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Windows compatibility
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---- Deterministic seed everywhere ----
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---- Paths ----
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ntn_telemetry_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHART_DIR = os.path.join(BASE_DIR, "charts")
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")

# ---- Feature columns (numeric only, no node_id/layer strings) ----
FEATURE_COLS = [
    "velocity_ms", "altitude_km", "doppler_hz", "rssi_dbm",
    "velocity_deviation", "gps_distance_km", "timestamp_delta_ms",
    "traffic_encoded", "layer_encoded",
]

# ---- Hyperparameters ----
N_ESTIMATORS = 200
CONTAMINATION = 0.15  # ~15% anomaly ratio matches our dataset (10K/50K = 20%)
MAX_SAMPLES = 256     # From the original Liu et al. 2008 paper


def load_data():
    """Load and prepare the dataset."""
    print("[1/6] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"      Loaded {len(df)} records")
    print(f"      Normal: {(df['is_spoofed'] == 0).sum()}, "
          f"Attack: {(df['is_spoofed'] == 1).sum()}")
    return df


def train_model(df):
    """Train Isolation Forest on normal data only."""
    print("[2/6] Training Isolation Forest model...")

    X = df[FEATURE_COLS].values
    y = df["is_spoofed"].values

    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train ONLY on normal data (unsupervised anomaly detection)
    normal_mask = y == 0
    X_train = X_scaled[normal_mask]
    print(f"      Training on {len(X_train)} normal records "
          f"(excluding {(~normal_mask).sum()} attack records)")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples=MAX_SAMPLES,
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use all CPU cores
    )
    model.fit(X_train)

    # Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"      Model saved to {MODEL_PATH}")
    print(f"      Scaler saved to {SCALER_PATH}")

    return model, scaler, X_scaled, y


def evaluate_model(model, scaler, X_scaled, y_true):
    """Evaluate model and compute metrics."""
    print("[3/6] Evaluating model performance...")

    # Predict: Isolation Forest returns 1=normal, -1=anomaly
    y_pred_raw = model.predict(X_scaled)
    # Convert: 1 -> 0 (normal), -1 -> 1 (anomaly) to match our label convention
    y_pred = (y_pred_raw == -1).astype(int)

    # Anomaly scores (lower = more anomalous)
    decision_scores = model.decision_function(X_scaled)
    # Invert so higher = more anomalous (for ROC curve)
    anomaly_scores = -decision_scores

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"      Accuracy:  {acc:.4f}")
    print(f"      F1 Score:  {f1:.4f}")

    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Attack"],
        digits=4,
    )
    print(f"\n{report}")

    return y_pred, anomaly_scores, report


def plot_confusion_matrix(y_true, y_pred):
    """Generate and save confusion matrix heatmap."""
    print("[4/6] Generating charts...")
    os.makedirs(CHART_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        annot_kws={"size": 16},
    )
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.title("Confusion Matrix - Isolation Forest Anomaly Detection", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("      -> confusion_matrix.png")


def plot_roc_curve(y_true, anomaly_scores):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#3b82f6", lw=2.5, label=f"Isolation Forest (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#64748b", lw=1.5, linestyle="--", label="Random Classifier")
    plt.fill_between(fpr, tpr, alpha=0.15, color="#3b82f6")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title(f"ROC Curve - NTN Anomaly Detection (AUC = {roc_auc:.4f})", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"      -> roc_curve.png (AUC = {roc_auc:.4f})")
    return roc_auc


def plot_precision_recall(y_true, anomaly_scores):
    """Generate and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
    ap = average_precision_score(y_true, anomaly_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#10b981", lw=2.5, label=f"Isolation Forest (AP = {ap:.4f})")
    plt.fill_between(recall, precision, alpha=0.15, color="#10b981")
    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.title(f"Precision-Recall Curve (AP = {ap:.4f})", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()
    print(f"      -> precision_recall_curve.png (AP = {ap:.4f})")


def plot_feature_importance(model, feature_names):
    """Plot feature importance based on average path length."""
    # Isolation Forest doesn't have direct feature_importances_,
    # so we compute it from average depth across all trees
    importances = np.zeros(len(feature_names))
    for tree in model.estimators_:
        importances += tree.feature_importances_
    importances /= len(model.estimators_)
    importances = importances / importances.sum() * 100  # normalize to percentages

    sorted_idx = np.argsort(importances)
    colors = sns.color_palette("viridis", len(feature_names))

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importances[sorted_idx],
             color=[colors[i] for i in range(len(feature_names))])
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx], fontsize=11)
    plt.xlabel("Relative Importance (%)", fontsize=13)
    plt.title("Feature Importance - Isolation Forest", fontsize=14)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("      -> feature_importance.png")


def plot_anomaly_distribution(y_true, anomaly_scores):
    """Plot distribution of anomaly scores for normal vs attack."""
    plt.figure(figsize=(10, 6))
    normal_scores = anomaly_scores[y_true == 0]
    attack_scores = anomaly_scores[y_true == 1]

    plt.hist(normal_scores, bins=80, alpha=0.65, color="#3b82f6",
             label=f"Normal (n={len(normal_scores)})", density=True)
    plt.hist(attack_scores, bins=80, alpha=0.65, color="#ef4444",
             label=f"Attack (n={len(attack_scores)})", density=True)
    plt.axvline(x=0, color="#f59e0b", linestyle="--", lw=2, label="Decision Boundary")
    plt.xlabel("Anomaly Score (higher = more anomalous)", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.title("Anomaly Score Distribution - Normal vs Attack", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "anomaly_score_distribution.png"), dpi=150)
    plt.close()
    print("      -> anomaly_score_distribution.png")


def save_report(report, roc_auc):
    """Save classification report to text file."""
    print("[5/6] Saving classification report...")
    report_path = os.path.join(CHART_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Isolation Forest - NTN Anomaly Detection Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:          Isolation Forest\n")
        f.write(f"n_estimators:   {N_ESTIMATORS}\n")
        f.write(f"contamination:  {CONTAMINATION}\n")
        f.write(f"max_samples:    {MAX_SAMPLES}\n")
        f.write(f"random_state:   {RANDOM_STATE}\n")
        f.write(f"ROC AUC:        {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 60 + "\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
    print(f"      -> classification_report.txt")


def main():
    print("=" * 60)
    print("  Isolation Forest Training Pipeline")
    print("  NTN Behavioural Zero-Trust Framework")
    print("=" * 60)
    print()

    df = load_data()
    model, scaler, X_scaled, y_true = train_model(df)
    y_pred, anomaly_scores, report = evaluate_model(model, scaler, X_scaled, y_true)

    plot_confusion_matrix(y_true, y_pred)
    roc_auc = plot_roc_curve(y_true, anomaly_scores)
    plot_precision_recall(y_true, anomaly_scores)
    plot_feature_importance(model, FEATURE_COLS)
    plot_anomaly_distribution(y_true, anomaly_scores)
    save_report(report, roc_auc)

    print()
    print("[6/6] Training pipeline complete!")
    print(f"      Model:  {MODEL_PATH}")
    print(f"      Charts: {CHART_DIR}/")
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  ROC AUC Score:  {roc_auc:.4f}")
    print(f"  Accuracy:       {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1 Score:       {f1_score(y_true, y_pred):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
