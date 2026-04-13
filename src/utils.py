# ============================================================
# utils.py — Logging, Config Loader, Helper Functions
# ============================================================

import logging
import os
import yaml
import numpy as np
from datetime import datetime


# ── Load YAML Config ────────────────────────────────────────
def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load the YAML configuration file.

    Args:
        config_path: Path to the config file.

    Returns:
        Dictionary with all config values.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {e}")


# ── Setup Logger ─────────────────────────────────────────────
def setup_logger(name: str = "predictive_maintenance",
                 log_file: str = "logs/pipeline.log",
                 level=logging.INFO) -> logging.Logger:
    """
    Set up a logger that writes to both console and file.

    Args:
        name: Logger name.
        log_file: Path to the log file.
        level: Logging level.

    Returns:
        Configured logger object.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ── Alert Classifier ─────────────────────────────────────────
def classify_alert(row: dict, thresholds: dict) -> str:
    """
    Classify sensor readings into alert levels.

    Args:
        row: Dictionary with sensor readings.
        thresholds: Threshold config dict.

    Returns:
        Alert level string: 'NORMAL', 'WARNING', or 'CRITICAL'
    """
    sensors = ["temperature", "vibration", "current", "pressure"]
    alert = "NORMAL"

    for sensor in sensors:
        if sensor not in row or sensor not in thresholds:
            continue
        value = row[sensor]
        t = thresholds[sensor]

        if value >= t["critical_max"]:
            return "CRITICAL"   # immediate return — highest severity
        elif value >= t["warning_max"]:
            alert = "WARNING"   # keep checking for CRITICAL

    return alert


# ── Maintenance Recommendation ───────────────────────────────
def get_maintenance_recommendation(alert_level: str,
                                   failure_prob: float,
                                   top_features: list) -> str:
    """
    Generate a human-readable maintenance recommendation.

    Args:
        alert_level: 'NORMAL', 'WARNING', or 'CRITICAL'
        failure_prob: Model probability of failure (0.0 – 1.0)
        top_features: List of feature names driving prediction.

    Returns:
        Recommendation string.
    """
    feature_str = ", ".join(top_features) if top_features else "multiple sensor readings"

    if alert_level == "CRITICAL":
        return (
            f"🚨 CRITICAL: Immediate shutdown recommended! "
            f"Failure probability: {failure_prob:.1%}. "
            f"Primary indicators: {feature_str}. "
            f"Schedule emergency maintenance NOW."
        )
    elif alert_level == "WARNING":
        return (
            f"⚠️  WARNING: Machine degradation detected. "
            f"Failure probability: {failure_prob:.1%}. "
            f"Primary indicators: {feature_str}. "
            f"Schedule maintenance within 24 hours."
        )
    else:
        return (
            f"✅ NORMAL: All sensors within acceptable range. "
            f"Failure probability: {failure_prob:.1%}. "
            f"Next scheduled maintenance due as planned."
        )


# ── Rolling Statistics ────────────────────────────────────────
def compute_rolling_stats(series: np.ndarray,
                           window: int = 10) -> tuple:
    """
    Compute rolling mean and standard deviation.

    Args:
        series: 1D numpy array.
        window: Rolling window size.

    Returns:
        Tuple of (rolling_mean, rolling_std) as numpy arrays.
    """
    n = len(series)
    rolling_mean = np.full(n, np.nan)
    rolling_std = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_vals = series[i - window + 1: i + 1]
        rolling_mean[i] = np.mean(window_vals)
        rolling_std[i] = np.std(window_vals)

    return rolling_mean, rolling_std


# ── Timestamp Generator ───────────────────────────────────────
def generate_timestamps(n: int, step_minutes: int = 10) -> list:
    """
    Generate a list of datetime timestamps at fixed intervals.

    Args:
        n: Number of timestamps.
        step_minutes: Minutes between each timestamp.

    Returns:
        List of datetime objects.
    """
    from datetime import timedelta
    start = datetime(2024, 1, 1, 0, 0, 0)
    return [start + timedelta(minutes=i * step_minutes) for i in range(n)]


# ── Pretty Print Metrics ──────────────────────────────────────
def print_metrics_table(metrics: dict) -> None:
    """
    Print model evaluation metrics in a clean table format.

    Args:
        metrics: Dictionary with model names as keys,
                 metric dicts as values.
    """
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 60)
    for model_name, m in metrics.items():
        print(
            f"{model_name:<25} "
            f"{m.get('accuracy', 0):.3f}  "
            f"{m.get('precision', 0):.3f}  "
            f"{m.get('recall', 0):.3f}  "
            f"{m.get('f1', 0):.3f}  "
            f"{m.get('roc_auc', 0):.3f}"
        )
    print("=" * 60 + "\n")
