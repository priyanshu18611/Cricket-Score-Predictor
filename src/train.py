"""
train.py
--------
Trains the T20 Cricket Score Prediction model.

Pipeline:
    1. Load raw match data (data/t20I_cricket_dataset.csv)
    2. Clean & rename columns to a standard schema
    3. Filter out cities / teams with very few matches (data consistency)
    4. One-Hot Encode categorical columns (batting_team, bowling_team, city)
    5. Train an XGBoost Regressor on an 80/20 train-test split
    6. Evaluate with R2 Score and Mean Absolute Error (MAE)
    7. Save the fitted pipeline to models/pipe.pkl

Run:
    python src/train.py
"""

import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "t20I_cricket_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pipe.pkl")

MIN_CITY_MATCHES = 600
MIN_TEAM_MATCHES = 1500


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "battingTeam": "batting_team",
            "bowlingTeam": "bowling_team",
            "city": "city",
            "score": "current_score",
            "delivery_left": "balls_left",
            "wicketsLeft": "wickets_left",
            "CurrentRunRate": "crr",
            "Run_In_Last5": "last_five",
            "Final_Score": "final_score",
        }
    )

    keep_cols = [
        "batting_team",
        "bowling_team",
        "city",
        "current_score",
        "balls_left",
        "wickets_left",
        "crr",
        "last_five",
        "final_score",
    ]
    df = df[keep_cols]
    df = df.dropna()
    df = df[(df["balls_left"] >= 0) & (df["wickets_left"] >= 0)]

    return df


def filter_rare_categories(df: pd.DataFrame) -> pd.DataFrame:
    city_counts = df["city"].value_counts()
    good_cities = city_counts[city_counts >= MIN_CITY_MATCHES].index
    df = df[df["city"].isin(good_cities)]

    team_counts = df["batting_team"].value_counts()
    good_teams = team_counts[team_counts >= MIN_TEAM_MATCHES].index
    df = df[df["batting_team"].isin(good_teams) & df["bowling_team"].isin(good_teams)]

    return df


def build_pipeline() -> Pipeline:
    trf = ColumnTransformer(
        [
            (
                "trf",
                OneHotEncoder(drop="first", sparse_output=False),
                ["batting_team", "bowling_team", "city"],
            )
        ],
        remainder="passthrough",
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        objective="reg:squarederror",
        random_state=42,
    )

    pipe = Pipeline(steps=[("step1", trf), ("step2", model)])
    return pipe


def main():
    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    df = filter_rare_categories(df)
    print(f"Final dataset shape after cleaning/filtering: {df.shape}")
    print(f"Teams kept: {sorted(df['batting_team'].unique())}")
    print(f"Cities kept: {sorted(df['city'].unique())}")

    feature_cols = [
        "batting_team",
        "bowling_team",
        "city",
        "current_score",
        "balls_left",
        "wickets_left",
        "crr",
        "last_five",
    ]
    X = df[feature_cols]
    y = df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training XGBoost Regressor...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f} runs")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
