"""
modelling.py

Melatih model machine learning untuk Fraud Detection
menggunakan MLflow Tracking (autolog).

Kriteria 2 - Basic:
- Menggunakan MLflow autolog.
- Menyimpan artefak model di MLflow Tracking UI lokal.
"""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def load_preprocessed_data(
    train_path: str = "fraud_detection_preprocessing/fraud_train_preprocessed.csv",
    test_path: str = "fraud_detection_preprocessing/fraud_test_preprocessed.csv",
    target_col: str = "Class",
):
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


def train_logreg_model():
    """Melatih Logistic Regression dengan MLflow autolog."""
    # 1. Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()

    # 2. Set MLflow tracking URI & experiment
    # Jika kamu menjalankan mlflow server di localhost:5000,
    # gunakan baris berikut:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("FraudDetection_Baseline")

    # 3. Aktifkan autolog
    mlflow.sklearn.autolog()

    # 4. Definisikan model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # 5. Mulai run MLflow
    with mlflow.start_run(run_name="logreg_baseline"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("[INFO] Training selesai. Classification report:")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    train_logreg_model()
