# ============================================================
# predict.py — Inference, Explainability & Alert Generation
# ============================================================
# Loads a saved model and scaler, runs predictions on new
# sensor readings, explains WHY a failure is predicted, and
# triggers the appropriate alert level.
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import (
    setup_logger, load_config,
    classify_alert, get_maintenance_recommendation
)
from src.preprocess import engineer_features, get_feature_columns

logger = setup_logger()


# ── Load Artefacts ────────────────────────────────────────────
def load_model_and_scaler(model_path: str,
                           scaler_path: str):
    """
    Load the saved sklearn model and scaler from disk.

    Args:
        model_path: Path to the .pkl model file.
        scaler_path: Path to the .pkl scaler file.

    Returns:
        Tuple of (model, scaler).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run main.py --train first."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}."
        )

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Scaler loaded from {scaler_path}")
    return model, scaler


# ── Explain Prediction ────────────────────────────────────────
def explain_prediction(model,
                        feature_names: list,
                        x_row: np.ndarray,
                        top_n: int = 3) -> list:
    """
    Identify the top contributing features for a prediction.

    Uses feature_importances_ for tree models,
    or absolute coefficient magnitudes for linear models.

    Args:
        model: Fitted sklearn estimator.
        feature_names: Feature column names.
        x_row: Single row feature vector (1D numpy array).
        top_n: Number of top features to return.

    Returns:
        List of top-N feature name strings.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return ["sensor readings"]

        top_idx = np.argsort(importances)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        return top_features

    except Exception as e:
        logger.warning(f"Could not explain prediction: {e}")
        return ["sensor readings"]


# ── Single-Row Prediction ─────────────────────────────────────
def predict_single(sensor_reading: dict,
                   model,
                   scaler,
                   feature_names: list,
                   thresholds: dict) -> dict:
    """
    Predict failure for a single sensor reading.

    Args:
        sensor_reading: Dict with keys:
                        temperature, vibration, current,
                        pressure, runtime_hours
        model: Fitted sklearn model.
        scaler: Fitted scaler.
        feature_names: Feature names expected by the model.
        thresholds: Sensor threshold config.

    Returns:
        Dict with prediction, probability, alert level,
        top features, and recommendation.
    """
    # Build a single-row DataFrame for feature engineering
    row_df = pd.DataFrame([sensor_reading])
    row_df["timestamp"] = pd.Timestamp.now()
    row_df["failure"]   = 0   # placeholder

    # Engineer features (requires >1 row for rolling — pad with duplicates)
    padded = pd.concat([row_df] * 12, ignore_index=True)
    padded = engineer_features(padded, window=10)
    row_feat = padded.iloc[[-1]][feature_names].values

    # Scale and predict
    row_scaled  = scaler.transform(row_feat)
    prediction  = model.predict(row_scaled)[0]
    failure_prob = model.predict_proba(row_scaled)[0][1]

    # Classify alert from raw sensor values
    alert_level = classify_alert(sensor_reading, thresholds)

    # Explain prediction
    top_features = explain_prediction(model, feature_names, row_scaled[0])

    # Generate recommendation
    recommendation = get_maintenance_recommendation(
        alert_level, failure_prob, top_features
    )

    return {
        "prediction":     int(prediction),
        "failure_prob":   round(float(failure_prob), 4),
        "alert_level":    alert_level,
        "top_features":   top_features,
        "recommendation": recommendation
    }


# ── Batch Prediction ──────────────────────────────────────────
def predict_batch(df: pd.DataFrame,
                  model,
                  scaler,
                  feature_names: list,
                  thresholds: dict,
                  save_path: str = "outputs/predictions.csv") -> pd.DataFrame:
    """
    Run predictions on an entire DataFrame.

    Args:
        df: Preprocessed (engineered) DataFrame.
        model: Fitted model.
        scaler: Fitted scaler.
        feature_names: Feature columns.
        thresholds: Sensor threshold config.
        save_path: Where to save results CSV.

    Returns:
        DataFrame with prediction columns added.
    """
    logger.info(f"Running batch prediction on {len(df)} samples...")

    X = scaler.transform(df[feature_names].values)
    df = df.copy()
    df["predicted_failure"] = model.predict(X)
    df["failure_prob"]      = model.predict_proba(X)[:, 1].round(4)

    # Alert level per row
    sensor_cols = ["temperature", "vibration", "current", "pressure"]
    df["alert_level"] = df[sensor_cols].apply(
        lambda row: classify_alert(row.to_dict(), thresholds), axis=1
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Batch predictions saved → {save_path}")

    detected = df["predicted_failure"].sum()
    logger.info(f"Failures detected: {detected} / {len(df)} "
                f"({detected/len(df):.1%})")

    return df


# ── Failure Timeline Plot ─────────────────────────────────────
def plot_failure_timeline(df: pd.DataFrame,
                          save_path: str = "outputs/graphs/failure_timeline.png"):
    """
    Plot actual vs predicted failures with failure probability
    overlaid as a heatmap timeline.

    Args:
        df: DataFrame with predictions.
        save_path: Where to save the plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              sharex=True, facecolor="#0f1117")

    x = range(len(df))
    colors_bg = "#0f1117"
    for ax in axes:
        ax.set_facecolor(colors_bg)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

    # Panel 1: Temperature + Vibration
    ax1 = axes[0]
    ax1.plot(x, df["temperature"], color="#FF7043", linewidth=0.8,
             label="Temperature (°C)", alpha=0.9)
    ax1.plot(x, df["vibration"] * 10, color="#42A5F5", linewidth=0.8,
             label="Vibration ×10 (mm/s)", alpha=0.9)
    ax1.set_ylabel("Sensor Value", color="white")
    ax1.legend(loc="upper left", fontsize=9, facecolor="#1e1e2e",
               labelcolor="white")
    ax1.set_title("AI-Powered Predictive Maintenance — Failure Timeline",
                  color="white", fontsize=14, fontweight="bold")

    # Panel 2: Failure Probability
    ax2 = axes[1]
    ax2.fill_between(x, df["failure_prob"], alpha=0.5,
                     color="#EF5350")
    ax2.plot(x, df["failure_prob"], color="#EF5350", linewidth=0.8)
    ax2.axhline(0.5, color="yellow", linestyle="--", linewidth=1,
                label="Decision threshold (0.5)")
    ax2.set_ylabel("Failure Prob.", color="white")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", fontsize=9,
               facecolor="#1e1e2e", labelcolor="white")

    # Panel 3: Actual vs Predicted Failure Events
    ax3 = axes[2]
    if "failure" in df.columns:
        ax3.scatter(
            [i for i in x if df["failure"].iloc[i] == 1],
            [1.1] * df["failure"].sum(),
            marker="^", color="#FF5722", s=40, label="Actual Failure", zorder=5
        )
    ax3.scatter(
        [i for i in x if df["predicted_failure"].iloc[i] == 1],
        [0.9] * df["predicted_failure"].sum(),
        marker="v", color="#FFEB3B", s=40, label="Predicted Failure", zorder=5
    )
    ax3.set_ylim(0.7, 1.3)
    ax3.set_yticks([])
    ax3.set_xlabel("Sample Index", color="white")
    ax3.set_ylabel("Events", color="white")
    ax3.legend(loc="upper left", fontsize=9,
               facecolor="#1e1e2e", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=colors_bg)
    plt.close()
    logger.info(f"Failure timeline saved → {save_path}")


# ── Sensor Trend Plot ─────────────────────────────────────────
def plot_sensor_trends(df: pd.DataFrame,
                       save_path: str = "outputs/graphs/sensor_trends.png"):
    """
    Four-panel sensor trend plot coloured by failure label.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sensors = ["temperature", "vibration", "current", "pressure"]
    labels_col = "failure" if "failure" in df.columns else "predicted_failure"

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Sensor Trends with Failure Events", fontsize=15,
                 fontweight="bold")
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        normal_mask  = df[labels_col] == 0
        failure_mask = df[labels_col] == 1

        ax.plot(df.index[normal_mask], df[sensor][normal_mask],
                color="#1E88E5", linewidth=0.7, label="Normal", alpha=0.7)
        ax.scatter(df.index[failure_mask], df[sensor][failure_mask],
                   color="#E53935", s=12, zorder=5, label="Failure")

        ax.set_title(sensor.capitalize(), fontsize=12)
        ax.set_xlabel("Sample")
        ax.set_ylabel(sensor)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Sensor trends plot saved → {save_path}")


if __name__ == "__main__":
    config  = load_config()
    model, scaler = load_model_and_scaler(
        config["paths"]["model_output"],
        config["paths"]["scaler_output"]
    )

    # Single prediction demo
    sample_input = {
        "temperature":  98.5,
        "vibration":    6.2,
        "current":      22.0,
        "pressure":     110.0,
        "runtime_hours": 950.0
    }

    from src.preprocess import engineer_features, get_feature_columns
    import pandas as pd
    demo_df = pd.read_csv(config["paths"]["processed_data"])
    feat_cols = get_feature_columns(demo_df)

    result = predict_single(
        sample_input, model, scaler,
        feat_cols, config["thresholds"]
    )

    print("\n" + "=" * 55)
    print("SINGLE PREDICTION RESULT")
    print("=" * 55)
    for k, v in result.items():
        print(f"  {k:<20}: {v}")
    print("=" * 55)
