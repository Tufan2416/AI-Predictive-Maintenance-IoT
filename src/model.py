# ============================================================
# model.py — Model Training, Evaluation & Selection
# ============================================================
# Trains Logistic Regression, Random Forest, and XGBoost.
# Handles class imbalance via SMOTE.
# Evaluates with Accuracy, Precision, Recall, F1, ROC-AUC.
# Saves the best model to disk.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

from src.utils import setup_logger, load_config, print_metrics_table

logger = setup_logger()


# ── Class-Imbalance Handler (manual SMOTE-lite) ───────────────
def oversample_minority(X: np.ndarray,
                         y: np.ndarray,
                         seed: int = 42) -> tuple:
    """
    Simple random oversampling of the minority (failure) class.
    Avoids hard dependency on imbalanced-learn while still
    balancing the dataset.

    Args:
        X: Feature matrix.
        y: Label array.
        seed: Random seed.

    Returns:
        Resampled (X_res, y_res).
    """
    np.random.seed(seed)
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]

    n_to_add = len(majority_idx) - len(minority_idx)
    if n_to_add <= 0:
        return X, y

    oversample_idx = np.random.choice(minority_idx,
                                       size=n_to_add,
                                       replace=True)
    X_res = np.vstack([X, X[oversample_idx]])
    y_res = np.concatenate([y, y[oversample_idx]])

    # Shuffle
    perm = np.random.permutation(len(y_res))
    return X_res[perm], y_res[perm]


# ── Build All Models ──────────────────────────────────────────
def build_models(random_state: int = 42) -> dict:
    """
    Instantiate all candidate models.

    Returns:
        Dictionary of model_name → estimator.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
    }

    # Optional XGBoost (gracefully skip if not installed)
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=5,   # handle imbalance
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0
        )
        logger.info("XGBoost detected and added to model pool.")
    except ImportError:
        logger.warning("XGBoost not installed — skipping. "
                       "Run: pip install xgboost")

    return models


# ── Evaluate One Model ────────────────────────────────────────
def evaluate_model(model, X_test: np.ndarray,
                   y_test: np.ndarray) -> dict:
    """
    Compute all evaluation metrics for a fitted model.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc.
    """
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "y_pred":    y_pred,
        "y_prob":    y_prob
    }


# ── Plot Confusion Matrix ─────────────────────────────────────
def plot_confusion_matrix(y_test: np.ndarray,
                           y_pred: np.ndarray,
                           model_name: str,
                           save_path: str = "outputs/graphs/confusion_matrix.png"):
    """Save a styled confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
        ax=ax, linewidths=0.5, linecolor="white"
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


# ── Plot ROC Curve ────────────────────────────────────────────
def plot_roc_curves(models_fitted: dict,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    save_path: str = "outputs/graphs/roc_curves.png"):
    """Plot and save ROC curves for all fitted models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
    for (name, model), color in zip(models_fitted.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved → {save_path}")


# ── Feature Importance ────────────────────────────────────────
def plot_feature_importance(model,
                             feature_names: list,
                             top_n: int = 15,
                             save_path: str = "outputs/graphs/feature_importance.png"):
    """Plot and save top-N feature importances (Random Forest / XGBoost)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature_importances_ attribute — skipping.")
        return

    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E53935" if i < 3 else "#1E88E5"
              for i in range(len(importances))]
    importances.plot(kind="barh", ax=ax, color=colors[::-1])
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved → {save_path}")


# ── Main Training Pipeline ────────────────────────────────────
def train_models(X: np.ndarray,
                 y: np.ndarray,
                 feature_names: list,
                 config: dict) -> tuple:
    """
    Full model training pipeline:
      1. Train/test split
      2. Oversample minority class
      3. Train all models
      4. Evaluate and compare
      5. Select and save the best model

    Args:
        X: Feature matrix (scaled).
        y: Labels.
        feature_names: Column names for feature plots.
        config: Config dict.

    Returns:
        Tuple of (best_model, metrics_dict, X_test, y_test).
    """
    model_cfg = config["model"]
    model_path = config["paths"]["model_output"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Train/test split (stratified — keeps failure ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_state"],
        stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape} | "
                f"Train failures: {y_train.sum()} | Test failures: {y_test.sum()}")

    # Oversample minority class in training set ONLY
    X_train_bal, y_train_bal = oversample_minority(
        X_train, y_train, seed=model_cfg["random_state"]
    )
    logger.info(f"After oversampling — Train: {X_train_bal.shape} "
                f"(failures: {y_train_bal.sum()})")

    # Train all models
    all_models  = build_models(random_state=model_cfg["random_state"])
    all_metrics = {}
    fitted      = {}

    for name, model in all_models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_bal, y_train_bal)
        metrics = evaluate_model(model, X_test, y_test)
        all_metrics[name] = metrics
        fitted[name] = model

        logger.info(
            f"  {name}: Acc={metrics['accuracy']:.3f} "
            f"Prec={metrics['precision']:.3f} "
            f"Rec={metrics['recall']:.3f} "
            f"F1={metrics['f1']:.3f} "
            f"AUC={metrics['roc_auc']:.3f}"
        )

    print_metrics_table(all_metrics)

    # Select best model by F1 score (best for imbalanced data)
    best_name = max(
        all_metrics, key=lambda k: all_metrics[k]["f1"]
    )
    best_model = fitted[best_name]
    logger.info(f"✅ Best model selected: {best_name} "
                f"(F1={all_metrics[best_name]['f1']:.3f})")

    # Save best model
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved → {model_path}")

    # Generate all evaluation plots
    best_metrics = all_metrics[best_name]
    plot_confusion_matrix(
        y_test, best_metrics["y_pred"], best_name,
        save_path="outputs/graphs/confusion_matrix.png"
    )
    plot_roc_curves(
        fitted, X_test, y_test,
        save_path="outputs/graphs/roc_curves.png"
    )
    plot_feature_importance(
        best_model, feature_names,
        save_path="outputs/graphs/feature_importance.png"
    )

    # Print detailed classification report
    print("\n" + "=" * 50)
    print(f"Classification Report — {best_name}")
    print("=" * 50)
    print(classification_report(
        y_test, best_metrics["y_pred"],
        target_names=["No Failure", "Failure"]
    ))

    return best_model, all_metrics, X_test, y_test


if __name__ == "__main__":
    from src.data_loader  import load_or_generate_data
    from src.preprocess   import run_preprocessing, get_feature_columns, scale_features

    config    = load_config()
    raw_df    = load_or_generate_data(config)
    proc_df   = run_preprocessing(raw_df, config)
    feat_cols = get_feature_columns(proc_df)

    X = proc_df[feat_cols].values
    y = proc_df["failure"].values

    X_sc, _, scaler = scale_features(
        X, X, config["paths"]["scaler_output"]
    )
    train_models(X_sc, y, feat_cols, config)
