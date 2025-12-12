import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# fit a scaler on X_train, apply to both splits, save scaled arrays and scaler object

def main(processed_dir):
    X_train_path = os.path.join(processed_dir, "X_train.csv")
    X_test_path = os.path.join(processed_dir, "X_test.csv")

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled_df.to_csv(os.path.join(processed_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(processed_dir, "X_test_scaled.csv"), index=False)

    # optional: save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    args = parser.parse_args()
    main(args.processed_dir)
