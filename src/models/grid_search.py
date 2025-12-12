import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# Choose a regression model, e.g. RandomForestRegressor, perform GridSearchCV, save best_params.pkl in models
# 
# https://mlops-guide.github.io/Versionamento/pipelines_dvc/
# 


def main(processed_dir, models_dir):
    os.makedirs(models_dir, exist_ok=True)

    X_train = pd.read_csv(os.path.join(processed_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    joblib.dump(best_params, os.path.join(models_dir, "best_params.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--models_dir", default="models")
    args = parser.parse_args()
    main(args.processed_dir, args.models_dir)
