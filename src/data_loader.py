# ============================================================
# data_loader.py — Synthetic IoT Sensor Data Generator
# ============================================================
# This module simulates real-world industrial machine sensor
# data including: normal behavior, gradual degradation,
# and sudden failure spikes — just like a real SCADA system.
# ============================================================

import numpy as np
import pandas as pd
import os
from src.utils import setup_logger, load_config, generate_timestamps

logger = setup_logger()


def generate_sensor_data(n_samples: int = 5000,
                         failure_rate: float = 0.15,
                         random_seed: int = 42,
                         step_minutes: int = 10) -> pd.DataFrame:
    """
    Simulate IoT sensor readings for an industrial machine.

    Simulates three machine states:
      - Normal Operation  (~85% of data)
      - Gradual Degradation (leading up to failure)
      - Sudden Failure Spikes (actual failure events)

    Sensor channels:
      - temperature  (°C)
      - vibration    (mm/s RMS)
      - current      (Amperes)
      - pressure     (PSI)
      - runtime_hours (cumulative operating hours)

    Args:
        n_samples: Total number of data points.
        failure_rate: Fraction of samples that are failure events.
        random_seed: For reproducibility.
        step_minutes: Time gap between each sample.

    Returns:
        DataFrame with sensor columns + 'failure' label.
    """
    np.random.seed(random_seed)
    logger.info(f"Generating {n_samples} synthetic sensor samples "
                f"(failure_rate={failure_rate:.0%})")

    timestamps = generate_timestamps(n_samples, step_minutes)

    # ── Base signals (Normal operation) ──────────────────────
    temperature = np.random.normal(loc=65, scale=5, size=n_samples)    # °C
    vibration   = np.random.normal(loc=2.0, scale=0.4, size=n_samples) # mm/s
    current     = np.random.normal(loc=12.0, scale=1.5, size=n_samples)# Amps
    pressure    = np.random.normal(loc=65, scale=5, size=n_samples)    # PSI
    runtime     = np.cumsum(np.full(n_samples, step_minutes / 60))     # hours

    # ── Simulate gradual degradation trend ───────────────────
    # Every ~500 samples a slow drift increases sensor values
    degradation_trend = np.zeros(n_samples)
    block_size = 500
    for i in range(0, n_samples, block_size):
        end = min(i + block_size, n_samples)
        ramp = np.linspace(0, np.random.uniform(0.5, 2.5), end - i)
        degradation_trend[i:end] += ramp

    temperature += degradation_trend * 1.8
    vibration   += degradation_trend * 0.3
    current     += degradation_trend * 0.5
    pressure    += degradation_trend * 0.9

    # ── Inject failure events ─────────────────────────────────
    failure_labels = np.zeros(n_samples, dtype=int)
    n_failures = int(n_samples * failure_rate)

    # Randomly pick failure indices (avoid first 100 samples)
    failure_indices = np.random.choice(
        np.arange(100, n_samples), size=n_failures, replace=False
    )

    for idx in failure_indices:
        failure_labels[idx] = 1

        # Each failure event: spike sensors well above normal
        temperature[idx] += np.random.uniform(25, 45)   # sharp temp rise
        vibration[idx]   += np.random.uniform(4.0, 8.0) # severe shaking
        current[idx]     += np.random.uniform(8.0, 15.0)# overcurrent
        pressure[idx]    += np.random.uniform(20, 40)   # pressure surge

        # Also elevate the 1–3 samples BEFORE the failure
        # (degradation leading up to failure — very realistic)
        pre_window = range(max(0, idx - 3), idx)
        for pre_idx in pre_window:
            temperature[pre_idx] += np.random.uniform(5, 15)
            vibration[pre_idx]   += np.random.uniform(1.0, 3.0)
            current[pre_idx]     += np.random.uniform(2.0, 5.0)
            pressure[pre_idx]    += np.random.uniform(5, 15)

    # ── Add realistic sensor noise ────────────────────────────
    temperature += np.random.normal(0, 1.5, n_samples)
    vibration   += np.random.normal(0, 0.1, n_samples)
    current     += np.random.normal(0, 0.3, n_samples)
    pressure    += np.random.normal(0, 2.0, n_samples)

    # ── Clip to physically realistic ranges ───────────────────
    temperature = np.clip(temperature, 20, 150)
    vibration   = np.clip(vibration,   0,  15)
    current     = np.clip(current,     0,  40)
    pressure    = np.clip(pressure,    0, 160)

    # ── Assemble DataFrame ────────────────────────────────────
    df = pd.DataFrame({
        "timestamp":    timestamps,
        "temperature":  np.round(temperature, 2),
        "vibration":    np.round(vibration, 3),
        "current":      np.round(current, 2),
        "pressure":     np.round(pressure, 2),
        "runtime_hours":np.round(runtime, 2),
        "failure":      failure_labels
    })

    logger.info(f"Dataset generated: {df.shape[0]} rows, "
                f"{df['failure'].sum()} failure events "
                f"({df['failure'].mean():.1%})")
    return df


def load_or_generate_data(config: dict) -> pd.DataFrame:
    """
    Load existing dataset from disk or generate a new one.

    Args:
        config: Full config dictionary.

    Returns:
        Sensor DataFrame.
    """
    raw_path = config["paths"]["raw_data"]
    sim_cfg   = config["simulation"]

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if os.path.exists(raw_path):
        logger.info(f"Loading existing dataset from {raw_path}")
        df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    else:
        logger.info("No dataset found — generating synthetic data...")
        df = generate_sensor_data(
            n_samples=sim_cfg["n_samples"],
            failure_rate=sim_cfg["failure_rate"],
            random_seed=sim_cfg["random_seed"],
            step_minutes=sim_cfg["time_step_minutes"]
        )
        df.to_csv(raw_path, index=False)
        logger.info(f"Dataset saved to {raw_path}")

    return df


if __name__ == "__main__":
    # Quick test
    config = load_config()
    df = load_or_generate_data(config)
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Failure rate: {df['failure'].mean():.2%}")
    print(df.describe())
