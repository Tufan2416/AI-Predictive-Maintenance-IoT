# ============================================================
# main.py — Master Pipeline Orchestrator
# ============================================================
# Run:  python main.py            → full pipeline (train + predict)
#       python main.py --train    → training only
#       python main.py --predict  → inference on saved model
#       python main.py --demo     → single-sensor demo prediction
# ============================================================

import argparse
import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.utils        import setup_logger, load_config
from src.data_loader  import load_or_generate_data
from src.preprocess   import run_preprocessing, get_feature_columns, scale_features
from src.model        import train_models
from src.predict      import (
    load_model_and_scaler, predict_batch,
    predict_single, plot_failure_timeline, plot_sensor_trends
)
from src.alert_system import generate_alerts, export_alert_log

logger = setup_logger()


# ── Banner ────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   AI-POWERED PREDICTIVE MAINTENANCE SYSTEM FOR IoT DEVICES  ║
║                  Built for Industry · GitHub Portfolio       ║
╚══════════════════════════════════════════════════════════════╝
"""


# ── Training Phase ────────────────────────────────────────────
def run_training_pipeline(config: dict):
    """Execute the full data → model training pipeline."""
    logger.info("=" * 55)
    logger.info("PHASE 1: LOAD / GENERATE DATA")
    logger.info("=" * 55)
    raw_df = load_or_generate_data(config)

    logger.info("=" * 55)
    logger.info("PHASE 2: PREPROCESSING & FEATURE ENGINEERING")
    logger.info("=" * 55)
    proc_df = run_preprocessing(raw_df, config)

    logger.info("=" * 55)
    logger.info("PHASE 3: MODEL TRAINING & EVALUATION")
    logger.info("=" * 55)
    feat_cols = get_feature_columns(proc_df)
    X = proc_df[feat_cols].values
    y = proc_df["failure"].values

    # Scale — fit on all data for training demo
    # (in production: fit only on train split, which model.py handles internally)
    X_scaled, _, scaler = scale_features(
        X, X,
        scaler_path=config["paths"]["scaler_output"]
    )

    best_model, metrics, X_test, y_test = train_models(
        X_scaled, y, feat_cols, config
    )

    logger.info("=" * 55)
    logger.info("PHASE 4: GENERATE VISUALISATIONS")
    logger.info("=" * 55)
    plot_sensor_trends(proc_df, save_path="outputs/graphs/sensor_trends.png")

    # Run batch predictions on full dataset for timeline plot
    proc_df_pred = predict_batch(
        proc_df, best_model, scaler,
        feat_cols, config["thresholds"],
        save_path="outputs/predictions.csv"
    )
    plot_failure_timeline(
        proc_df_pred,
        save_path="outputs/graphs/failure_timeline.png"
    )

    logger.info("=" * 55)
    logger.info("PHASE 5: ALERT GENERATION")
    logger.info("=" * 55)
    alerts = generate_alerts(proc_df_pred, config["thresholds"])
    export_alert_log(alerts)

    logger.info("\n✅ Training pipeline complete!")
    logger.info("Outputs saved in: outputs/graphs/ | outputs/reports/")
    logger.info("Run 'streamlit run dashboard/app.py' to launch dashboard.")
    return best_model, scaler, proc_df, feat_cols


# ── Inference Phase ───────────────────────────────────────────
def run_inference_pipeline(config: dict):
    """Load saved model and run batch predictions."""
    model, scaler = load_model_and_scaler(
        config["paths"]["model_output"],
        config["paths"]["scaler_output"]
    )

    proc_path = config["paths"]["processed_data"]
    if not os.path.exists(proc_path):
        raise FileNotFoundError(
            "Processed data not found. Run --train first."
        )

    proc_df   = pd.read_csv(proc_path, parse_dates=["timestamp"])
    feat_cols = get_feature_columns(proc_df)

    pred_df = predict_batch(
        proc_df, model, scaler,
        feat_cols, config["thresholds"]
    )
    plot_failure_timeline(
        pred_df,
        save_path="outputs/graphs/failure_timeline.png"
    )
    alerts = generate_alerts(pred_df, config["thresholds"])
    export_alert_log(alerts)
    logger.info("✅ Inference pipeline complete.")


# ── Single Demo Prediction ────────────────────────────────────
def run_demo_prediction(config: dict):
    """Demonstrate a single-sensor reading prediction."""
    model, scaler = load_model_and_scaler(
        config["paths"]["model_output"],
        config["paths"]["scaler_output"]
    )

    proc_df   = pd.read_csv(config["paths"]["processed_data"],
                             parse_dates=["timestamp"])
    feat_cols = get_feature_columns(proc_df)

    # Simulate two scenarios
    scenarios = [
        {
            "name": "Healthy Machine",
            "reading": {
                "temperature": 68.0,
                "vibration":   2.1,
                "current":     12.5,
                "pressure":    67.0,
                "runtime_hours": 200.0
            }
        },
        {
            "name": "Machine Under Stress (Pre-Failure)",
            "reading": {
                "temperature": 97.5,
                "vibration":   6.8,
                "current":     22.0,
                "pressure":    108.0,
                "runtime_hours": 960.0
            }
        }
    ]

    print("\n" + "=" * 60)
    print("  PREDICTIVE MAINTENANCE — DEMO PREDICTIONS")
    print("=" * 60)

    for scenario in scenarios:
        result = predict_single(
            scenario["reading"], model, scaler,
            feat_cols, config["thresholds"]
        )

        status_icon = "🚨" if result["alert_level"] == "CRITICAL" else \
                      "⚠️ " if result["alert_level"] == "WARNING" else "✅"

        print(f"\n📟 Scenario: {scenario['name']}")
        print(f"   Sensor Inputs   : {scenario['reading']}")
        print(f"   Prediction      : {'⚡ FAILURE' if result['prediction'] else '✅ NO FAILURE'}")
        print(f"   Failure Prob    : {result['failure_prob']:.1%}")
        print(f"   Alert Level     : {status_icon} {result['alert_level']}")
        print(f"   Top Features    : {', '.join(result['top_features'])}")
        print(f"   Recommendation  : {result['recommendation']}")

    print("\n" + "=" * 60)


# ── Entry Point ───────────────────────────────────────────────
def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="AI Predictive Maintenance Pipeline"
    )
    parser.add_argument("--train",   action="store_true",
                        help="Run full training pipeline")
    parser.add_argument("--predict", action="store_true",
                        help="Run inference on saved model")
    parser.add_argument("--demo",    action="store_true",
                        help="Run single-sensor demo prediction")
    args = parser.parse_args()

    config = load_config()
    os.makedirs("logs", exist_ok=True)

    if args.train:
        run_training_pipeline(config)
    elif args.predict:
        run_inference_pipeline(config)
    elif args.demo:
        run_demo_prediction(config)
    else:
        # Default: run full pipeline
        logger.info("No flag specified — running FULL PIPELINE")
        run_training_pipeline(config)
        run_demo_prediction(config)


if __name__ == "__main__":
    main()
