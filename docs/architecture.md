# 🏗️ System Architecture — AI Predictive Maintenance

## Data Flow

```
[IoT Sensors / Synthetic Generator]
         │
         ▼
[data_loader.py]  ─────────────── generates 5000 sensor rows
   temperature, vibration,
   current, pressure, runtime
         │
         ▼
[preprocess.py]   ─────────────── clean → rolling stats → delta features
   rolling_mean, rolling_std,
   delta, deviation, stress_score,
   anomaly flags, time features
         │
         ▼
[model.py]        ─────────────── train 3 models, select best by F1
   Logistic Regression
   Random Forest  ← selected
   XGBoost
         │
         ▼
[predict.py]      ─────────────── batch + single inference
   failure_prob, prediction,
   feature attribution
         │
         ▼
[alert_system.py] ─────────────── threshold classification + recommendations
   NORMAL / WARNING / CRITICAL
   auto-shutdown check
         │
         ▼
[dashboard/app.py]─────────────── Streamlit live dashboard
   live KPIs, trend charts,
   probability gauge, alert badge
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `data_loader.py` | Synthetic sensor data generation with realistic failure patterns |
| `preprocess.py` | Data cleaning, rolling statistics, feature engineering |
| `model.py` | Multi-model training, evaluation, best model selection |
| `predict.py` | Single + batch inference, explainability, visualization |
| `alert_system.py` | Alert classification, recommendations, auto-shutdown |
| `utils.py` | Logging, config loading, shared helpers |
| `dashboard/app.py` | Streamlit monitoring dashboard with Plotly charts |
| `main.py` | CLI pipeline orchestrator |

## Key Design Decisions

1. **Synthetic data over static dataset** — gives full control over failure rate, 
   degradation patterns, and sensor behaviour. More educational and demonstrable.

2. **Random Forest as default best model** — balances performance and interpretability.
   Feature importances are native to the model.

3. **Three-tier alert system** — mirrors real industrial SCADA systems (green/amber/red).

4. **Oversampling over undersampling** — preserves all failure events for training
   (failure data is precious and limited).

5. **Modular pipeline** — each phase can be run independently, mirroring
   production ML systems.
