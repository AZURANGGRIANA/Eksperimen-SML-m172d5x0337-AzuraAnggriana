import os
import warnings
from typing import Tuple, Dict, Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore")

# 1. Load dataset train & test hasil preprocessing
def load_preprocessed_data(
    train_path: str = "fraud_detection_preprocessing/fraud_train_preprocessed.csv",
    test_path: str = "fraud_detection_preprocessing/fraud_test_preprocessed.csv",
    target_col: str = "Class",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Memuat dataset train dan test hasil preprocessing."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File train tidak ditemukan: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"File test tidak ditemukan: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    print("[INFO] Data train & test berhasil dimuat.")
    print(f"[INFO] X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test

# 2. Helper untuk membuat & log plot sebagai artifact
def log_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, filename: str = "training_confusion_matrix.png"
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Normalized confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    mlflow.log_artifact(filename)
    os.remove(filename)


def log_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, filename: str = "training_roc_curve.png"
) -> None:
    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, name="ROC")
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    mlflow.log_artifact(filename)
    os.remove(filename)


def log_precision_recall_curve(
    y_true: np.ndarray, y_proba: np.ndarray, filename: str = "training_precision_recall_curve.png"
) -> None:
    plt.figure(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, name="PR")
    plt.title("Precision recall curve")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    mlflow.log_artifact(filename)
    os.remove(filename)

# 3. Fungsi untuk train + evaluasi + logging manual ke MLflow
def train_and_log_model(
    model_name: str,
    estimator,
    param_distributions: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int = 15,
) -> None:
    print(f"\n[INFO] Mulai tuning untuk model: {model_name}")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print(f"[INFO] Best params ({model_name}): {search.best_params_}")
    print(f"[INFO] Best CV ROC-AUC ({model_name}): {search.best_score_:.4f}")

    # Prediksi
    y_pred = best_model.predict(X_test)

    # Kalau model punya predict_proba, pakai probabilitas kelas positif
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
    else:
        # fallback: decision_function -> diubah ke [0,1] via min-max
        scores = best_model.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min())

    # Hitung metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)

    print(f"[METRIC] {model_name} | acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  roc_auc={roc:.4f}")

    # MLflow logging (manual)
    with mlflow.start_run(run_name=f"{model_name}_tuning"):
        # params = best params dari tuning
        mlflow.log_params(search.best_params_)

        # metrics
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc)

        # log plots sebagai artifact
        log_confusion_matrix(y_test, y_pred)
        log_roc_curve(y_test, y_proba)
        log_precision_recall_curve(y_test, y_proba)

        # log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print(f"[INFO] Run MLflow untuk {model_name} selesai. Run ID: {mlflow.active_run().info.run_id}")

# 4. Pipeline utama tuning
def run_tuning_pipeline() -> None:
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()

    # Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("FraudDetection_Tuning")

    # Definisi model & search space
    models: Dict[str, Dict[str, Any]] = {
        "logreg": {
            "estimator": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="saga",
                n_jobs=-1,
            ),
            "params": {
                "C": np.logspace(-3, 3, 20),
                "penalty": ["l1", "l2"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            "params": {
                "n_estimators": [100, 200, 300, 400],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        },
    }

    # Jalankan tuning untuk setiap model
    for name, cfg in models.items():
        train_and_log_model(
            model_name=name,
            estimator=cfg["estimator"],
            param_distributions=cfg["params"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

    print("\n[INFO] Semua model selesai dituning dan dilogging ke MLflow.")

# 5. Auto-run
if __name__ == "__main__":
    run_tuning_pipeline()
