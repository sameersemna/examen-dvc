import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# read data/raw/raw.csv, split into train/test, save 4 CSVs in data/processed

def main(raw_path, processed_dir, test_size=0.2, random_state=42):
    os.makedirs(processed_dir, exist_ok=True)
    df = pd.read_csv(raw_path)

    # Drop datetime column if present (usually first column)
    if df.columns[0].lower().startswith("date") or "date" in df.columns[0].lower():
        df = df.iloc[:, 1:]

    # target is last column: silica_concentrate
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", default="data/raw/raw.csv")
    parser.add_argument("--processed_dir", default="data/processed")
    args = parser.parse_args()
    main(args.raw_path, args.processed_dir)
