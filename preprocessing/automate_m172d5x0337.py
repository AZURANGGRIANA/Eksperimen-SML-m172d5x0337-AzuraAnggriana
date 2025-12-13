import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. Load dataset mentah
def load_raw_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    df = pd.read_csv(file_path)
    print(f"[INFO] Dataset berhasil dimuat. Shape: {df.shape}")
    return df


# 2. Preprocessing lengkap
def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42
):
    # Hapus duplikat
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    print(f"[INFO] Menghapus duplikat: {initial_shape[0] - final_shape[0]} baris dihapus.")
    print(f"[INFO] Shape setelah drop duplikat: {final_shape}")

    # Pisahkan fitur dan target
    if target_col not in df.columns:
        raise KeyError(f"Kolom target '{target_col}' tidak ditemukan.")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    print(f"[INFO] Shape X: {X.shape}")
    print(f"[INFO] Shape y: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"[INFO] X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[INFO] y_train: {y_train.shape}, y_test: {y_test.shape}")

    # Scaling
    scaler = StandardScaler()
    num_cols = X_train.columns

    X_train_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_scaled = scaler.transform(X_test[num_cols])

    # Kembalikan ke DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=num_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=num_cols, index=X_test.index)

    # Tambahkan target kembali
    train_preprocessed = pd.concat([X_train_scaled, y_train], axis=1)
    test_preprocessed = pd.concat([X_test_scaled, y_test], axis=1)

    return train_preprocessed, test_preprocessed, scaler


# 3. Simpan data hasil preprocess
def save_preprocessed_data(
    train_preprocessed: pd.DataFrame,
    test_preprocessed: pd.DataFrame,
    output_dir: str = "namadataset_preprocessing", 
    train_filename: str = "fraud_train_preprocessed.csv",
    test_filename: str = "fraud_test_preprocessed.csv"
):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, train_filename)
    test_path = os.path.join(output_dir, test_filename)

    train_preprocessed.to_csv(train_path, index=False)
    test_preprocessed.to_csv(test_path, index=False)

    print(f"[INFO] Train preprocessed disimpan di: {train_path}")
    print(f"[INFO] Test preprocessed disimpan di : {test_path}")


# 4. Pipeline utama
def run_pipeline(
    file_path: str = "fraud_detection_20k.csv",
    target_col: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "namadataset_preprocessing"
):
    print("[INFO] Memulai pipeline preprocessing...")

    df = load_raw_data(file_path)
    train_preprocessed, test_preprocessed, _ = preprocess_data(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state
    )

    save_preprocessed_data(
        train_preprocessed,
        test_preprocessed,
        output_dir=output_dir
    )

    print("[INFO] Pipeline preprocessing selesai.")


# 5. Auto-run jika file dieksekusi langsung
if __name__ == "__main__":
    run_pipeline()
