import mlflow
import pandas as pd
import argparse

def load_model(model_uri):
    print(f"[INFO] Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def predict(model, input_csv):
    df = pd.read_csv(input_csv)
    print("[INFO] Input:")
    print(df.head())

    preds = model.predict(df)
    print("[INFO] Prediction result:")
    print(preds)
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_uri",
        type=str,
        default="model",
        help="Path to saved MLflow model",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="CSV file containing data to predict",
    )

    args = parser.parse_args()

    model = load_model(args.model_uri)
    predict(model, args.input_csv)

if __name__ == "__main__":
    main()
