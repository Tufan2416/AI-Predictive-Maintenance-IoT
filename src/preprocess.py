# ============================================================
# preprocess.py — Data Cleaning & Feature Engineering
# ============================================================
# Transforms raw sensor readings into model-ready features.
# Includes rolling statistics, lag features, and anomaly flags
# that help the model detect degradation patterns.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.utils import setup_logger, load_config

logger = setup_logger()


# ── Cleaning ─────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, duplicates, and outlier clipping.

    Args:
        df: Raw sensor DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Starting data cleaning...")
    original_shape = df.shape

    # Drop duplicate timestamps
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Fill any NaNs with forward-fill then backward-fill
    sensor_cols = ["temperature", "vibration", "current", "pressure", "runtime_hours"]
    df[sensor_cols] = df[sensor_cols].ffill().bfill()

    # Clip extreme outliers (beyond 99.9th percentile for safety)
    for col in ["temperature", "vibration", "current", "pressure"]:
        upper = df[col].quantile(0.999)
        lower = df[col].quantile(0.001)
        df[col] = df[col].clip(lower, upper)

    logger.info(f"Cleaning done: {original_shape} → {df.shape}")
    return df


# ── Feature Engineering ───────────────────────────────────────
def engineer_features(df: pd.DataFrame,
                       window: int = 10) -> pd.DataFrame:
    """
    Add derived features that improve failure detection:
      - Rolling mean & std for each sensor
      - Rate of change (delta) — catches sudden spikes
      - Composite stress score
      - Time-based features (hour of day, day of week)
      - Anomaly flag per sensor

    Args:
        df: Cleaned DataFrame.
        window: Rolling window size (default=10 steps = 100 min).

    Returns:
        DataFrame enriched with new features.
    """
    logger.info(f"Engineering features (window={window})...")
    df = df.copy()
    sensors = ["temperature", "vibration", "current", "pressure"]

    for col in sensors:
        # Rolling mean — smoothed baseline
        df[f"{col}_roll_mean"] = (
            df[col].rolling(window=window, min_periods=1).mean()
        )
        # Rolling std — captures instability/fluctuation
        df[f"{col}_roll_std"] = (
            df[col].rolling(window=window, min_periods=1).std().fillna(0)
        )
        # Delta (rate of change) — catches sudden spikes
        df[f"{col}_delta"] = df[col].diff().fillna(0)
        # Deviation from rolling mean — normalized anomaly signal
        df[f"{col}_deviation"] = df[col] - df[f"{col}_roll_mean"]

    # ── Composite stress score ────────────────────────────────
    # Weighted sum of normalised sensors — higher = more stressed
    df["stress_score"] = (
        0.35 * (df["temperature"] / 105) +
        0.30 * (df["vibration"]   / 8.0) +
        0.20 * (df["current"]     / 25.0) +
        0.15 * (df["pressure"]    / 120)
    ).round(4)

    # ── Time-based features ───────────────────────────────────
    if "timestamp" in df.columns:
        df["hour_of_day"]  = pd.to_datetime(df["timestamp"]).dt.hour
        df["day_of_week"]  = pd.to_datetime(df["timestamp"]).dt.dayofweek

    # ── Anomaly flags per sensor ──────────────────────────────
    # Flag if current reading exceeds rolling mean by >2 std devs
    for col in sensors:
        df[f"{col}_anomaly"] = (
            (df[f"{col}_deviation"].abs() >
             2 * df[f"{col}_roll_std"].clip(lower=0.01)).astype(int)
        )

    # ── Runtime fatigue flag ──────────────────────────────────
    df["high_runtime"] = (df["runtime_hours"] > 800).astype(int)

    logger.info(f"Feature engineering complete. "
                f"Total features: {df.shape[1]}")
    return df


# ── Scale Features ────────────────────────────────────────────
def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   scaler_path: str = "models/scaler.pkl") -> tuple:
    """
    Fit StandardScaler on training data, transform both sets.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        scaler_path: Where to save the fitted scaler.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler).
    """
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler fitted and saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


# ── Get Feature Matrix ────────────────────────────────────────
def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return list of all feature columns (exclude metadata & label).

    Args:
        df: Fully engineered DataFrame.

    Returns:
        List of feature column names.
    """
    exclude = {"timestamp", "failure"}
    return [c for c in df.columns if c not in exclude]


# ── Full Pipeline ─────────────────────────────────────────────
def run_preprocessing(df: pd.DataFrame,
                      config: dict) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline: clean → engineer → save.

    Args:
        df: Raw sensor DataFrame.
        config: Config dictionary.

    Returns:
        Preprocessed DataFrame ready for modelling.
    """
    df = clean_data(df)
    df = engineer_features(df, window=10)

    processed_path = config["paths"]["processed_data"]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")

    return df


if __name__ == "__main__":
    from src.data_loader import load_or_generate_data
    config = load_config()
    raw_df = load_or_generate_data(config)
    processed_df = run_preprocessing(raw_df, config)
    print(processed_df.head())
    print(f"\nFeatures: {get_feature_columns(processed_df)}")
