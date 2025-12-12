import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load best parameters, train final model on full X_train_scaled, save model as .pkl in models

def main(processed_dir, models_dir):
    os.makedirs(models_dir, exist_ok=True)

    X_train = pd.read_csv(os.path.join(processed_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()

    best_params_path = os.path.join(models_dir, "best_params.pkl")
    best_params = joblib.load(best_params_path)

    model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(models_dir, "model.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--models_dir", default="models")
    args = parser.parse_args()
    main(args.processed_dir, args.models_dir)
