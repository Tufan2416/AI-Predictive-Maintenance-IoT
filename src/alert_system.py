# ============================================================
# alert_system.py — Alert Engine & Maintenance Scheduler
# ============================================================
# Processes predictions and generates structured alerts,
# maintenance recommendations, and optional auto-shutdown logic.
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime
from src.utils import setup_logger, load_config, classify_alert

logger = setup_logger()


# ── Alert Record ──────────────────────────────────────────────
class Alert:
    """Represents a single machine alert event."""

    def __init__(self, timestamp, level: str, failure_prob: float,
                 sensor_values: dict, recommendation: str):
        self.timestamp      = timestamp
        self.level          = level          # NORMAL | WARNING | CRITICAL
        self.failure_prob   = failure_prob
        self.sensor_values  = sensor_values
        self.recommendation = recommendation

    def to_dict(self) -> dict:
        return {
            "timestamp":     str(self.timestamp),
            "alert_level":   self.level,
            "failure_prob":  round(self.failure_prob, 4),
            **self.sensor_values,
            "recommendation": self.recommendation
        }

    def __repr__(self):
        return (
            f"[{self.timestamp}] {self.level} "
            f"(prob={self.failure_prob:.2%}) — {self.recommendation[:60]}..."
        )


# ── Generate Alerts from Batch Predictions ────────────────────
def generate_alerts(pred_df: pd.DataFrame,
                    thresholds: dict,
                    min_prob: float = 0.4) -> list:
    """
    Scan batch prediction DataFrame and emit Alert objects
    for any WARNING or CRITICAL readings.

    Args:
        pred_df: DataFrame with failure_prob, alert_level columns.
        thresholds: Sensor threshold config.
        min_prob: Minimum failure probability to trigger alert.

    Returns:
        List of Alert objects sorted by timestamp.
    """
    alerts = []
    sensor_cols = ["temperature", "vibration", "current", "pressure"]

    for _, row in pred_df.iterrows():
        level = row.get("alert_level", "NORMAL")
        prob  = row.get("failure_prob", 0.0)

        if level in ("WARNING", "CRITICAL") or prob >= min_prob:
            recommendation = _make_recommendation(level, prob, row, sensor_cols, thresholds)
            alert = Alert(
                timestamp      = row.get("timestamp", datetime.now()),
                level          = level,
                failure_prob   = prob,
                sensor_values  = {c: round(row[c], 2)
                                  for c in sensor_cols if c in row},
                recommendation = recommendation
            )
            alerts.append(alert)

    logger.info(f"Alert scan complete: {len(alerts)} alerts "
                f"({sum(1 for a in alerts if a.level=='CRITICAL')} CRITICAL, "
                f"{sum(1 for a in alerts if a.level=='WARNING')} WARNING)")
    return alerts


def _make_recommendation(level: str, prob: float,
                          row: pd.Series,
                          sensor_cols: list,
                          thresholds: dict) -> str:
    """Build a specific recommendation based on which sensors are elevated."""
    offenders = []
    for col in sensor_cols:
        if col not in row or col not in thresholds:
            continue
        val = row[col]
        t   = thresholds[col]
        if val >= t["critical_max"]:
            offenders.append(f"{col}={val:.1f} (CRITICAL)")
        elif val >= t["warning_max"]:
            offenders.append(f"{col}={val:.1f} (HIGH)")

    sensor_summary = "; ".join(offenders) if offenders else "multiple sensors elevated"

    if level == "CRITICAL":
        return (
            f"IMMEDIATE ACTION REQUIRED. "
            f"Failure probability {prob:.1%}. Elevated: {sensor_summary}. "
            f"Initiate emergency shutdown protocol."
        )
    elif level == "WARNING":
        return (
            f"Schedule maintenance within 24 hrs. "
            f"Failure probability {prob:.1%}. Elevated: {sensor_summary}."
        )
    else:
        return f"Monitor closely. Failure probability {prob:.1%}."


# ── Auto-Shutdown Logic ───────────────────────────────────────
def auto_shutdown_check(sensor_reading: dict,
                         failure_prob: float,
                         thresholds: dict,
                         shutdown_prob_threshold: float = 0.85) -> bool:
    """
    Simulate auto-shutdown decision.

    Machine is flagged for shutdown if:
      - failure probability exceeds shutdown_prob_threshold, OR
      - any single sensor exceeds its CRITICAL threshold.

    Args:
        sensor_reading: Current sensor values dict.
        failure_prob: Model failure probability.
        thresholds: Threshold config.
        shutdown_prob_threshold: Prob above which to shutdown.

    Returns:
        True if machine should be shut down.
    """
    if failure_prob >= shutdown_prob_threshold:
        logger.warning(
            f"AUTO-SHUTDOWN TRIGGERED: failure_prob={failure_prob:.2%} "
            f"exceeds threshold {shutdown_prob_threshold:.0%}"
        )
        return True

    for sensor, t in thresholds.items():
        if sensor not in sensor_reading:
            continue
        if sensor_reading[sensor] >= t["critical_max"]:
            logger.warning(
                f"AUTO-SHUTDOWN TRIGGERED: {sensor}={sensor_reading[sensor]} "
                f"exceeds critical max {t['critical_max']}"
            )
            return True

    return False


# ── Export Alert Log ──────────────────────────────────────────
def export_alert_log(alerts: list,
                      save_path: str = "outputs/reports/alert_log.csv"):
    """Save all alerts to a CSV report."""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not alerts:
        logger.info("No alerts to export.")
        return

    records = [a.to_dict() for a in alerts]
    pd.DataFrame(records).to_csv(save_path, index=False)
    logger.info(f"Alert log exported → {save_path} ({len(alerts)} records)")


if __name__ == "__main__":
    # Demo: generate fake prediction rows and test alert engine
    config = load_config()
    np.random.seed(42)

    dummy_data = {
        "timestamp":    pd.date_range("2024-01-01", periods=20, freq="10min"),
        "temperature":  np.random.uniform(60, 120, 20),
        "vibration":    np.random.uniform(1.5, 9.0, 20),
        "current":      np.random.uniform(10, 28, 20),
        "pressure":     np.random.uniform(60, 130, 20),
        "failure_prob": np.random.uniform(0.0, 1.0, 20),
        "alert_level":  np.random.choice(
            ["NORMAL", "WARNING", "CRITICAL"], 20,
            p=[0.6, 0.3, 0.1]
        )
    }
    df = pd.DataFrame(dummy_data)
    alerts = generate_alerts(df, config["thresholds"])
    for a in alerts[:5]:
        print(a)
    export_alert_log(alerts)
