import argparse
import json
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# use model, make predictions on X_test_scaled, save predictions CSV and metrics JSON
# https://mlops-guide.github.io/Versionamento/pipelines_dvc/

def main(processed_dir, models_dir, metrics_dir, data_dir):
    os.makedirs(metrics_dir, exist_ok=True)

    X_test = pd.read_csv(os.path.join(processed_dir, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.ravel()

    model = joblib.load(os.path.join(models_dir, "model.pkl"))
    y_pred = model.predict(X_test)

    # Save predictions
    os.makedirs(data_dir, exist_ok=True)
    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    pred_df.to_csv(os.path.join(data_dir, "predictions.csv"), index=False)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mse": mse, "r2": r2}
    with open(os.path.join(metrics_dir, "scores.json"), "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--metrics_dir", default="metrics")
    parser.add_argument(
        "--data_dir", default="data", help="directory where predictions.csv will be saved"
    )
    args = parser.parse_args()
    main(args.processed_dir, args.models_dir, args.metrics_dir, args.data_dir)
