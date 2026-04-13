# 📅 7-Day GitHub Commit Plan & Proof Checklist

## Day-by-Day Commit Strategy

---

### Day 1 — Project Setup & Structure

**Actions:**
1. Create GitHub repo: `AI-Predictive-Maintenance-IoT`
2. Description: `AI system that predicts industrial machine failures using IoT sensor data — ML pipeline with Streamlit dashboard`
3. Add topics: `machine-learning`, `iot`, `predictive-maintenance`, `python`, `streamlit`, `random-forest`, `time-series`

```bash
git init
git remote add origin https://github.com/tufan2416/AI-Predictive-Maintenance-IoT.git
git add .gitignore requirements.txt README.md config/
git commit -m "chore: initial project setup, config, and requirements"
git push -u origin main
```

**Proof:** Screenshot of GitHub repo with topic tags visible.

---

### Day 2 — Data Simulation

```bash
git add src/data_loader.py src/utils.py src/__init__.py
git commit -m "feat: synthetic IoT sensor data generator with failure patterns"
git push
```

**What to capture:**
- Run `python -c "from src.data_loader import *; config=load_config(); df=load_or_generate_data(config); print(df.head(10)); print(df.describe())"`
- Screenshot the terminal output showing data shape + failure rate
- Screenshot `data/raw/sensor_data.csv` opened in VS Code

---

### Day 3 — Preprocessing & Feature Engineering

```bash
git add src/preprocess.py
git commit -m "feat: data cleaning + 20+ engineered features (rolling, delta, stress score)"
git push
```

**What to capture:**
- Print feature column list (show 20+ features created)
- Show before vs after shape of DataFrame

---

### Day 4 — Model Training

```bash
git add src/model.py
git commit -m "feat: train LR + Random Forest + XGBoost with class balancing"
git push
```

**What to capture:**
- Screenshot of training log showing all 3 model metrics table
- Screenshot of `outputs/graphs/confusion_matrix.png`
- Screenshot of `outputs/graphs/roc_curves.png`
- Screenshot of `outputs/graphs/feature_importance.png`

---

### Day 5 — Prediction & Alert System

```bash
git add src/predict.py src/alert_system.py
git commit -m "feat: failure prediction engine with explainability + 3-tier alert system"
git push
```

**What to capture:**
- Run `python main.py --demo` — screenshot both scenario outputs
- Show CRITICAL alert output in terminal

---

### Day 6 — Dashboard & Visualizations

```bash
git add dashboard/ outputs/graphs/
git commit -m "feat: Streamlit live monitoring dashboard with Plotly charts"
git push
```

**What to capture:**
- Screenshot of dashboard running at `localhost:8501`
- Show CRITICAL mode with red alert badge
- Show sensor trend chart

---

### Day 7 — Final Polish & Documentation

```bash
git add notebooks/ docs/ logs/ main.py
git commit -m "docs: EDA notebook, architecture docs, pipeline runner"
git push
```

**Final GitHub repo checklist:**
- [ ] Professional README with badges
- [ ] All 5 output graphs in `outputs/graphs/`
- [ ] Notebook with EDA
- [ ] `requirements.txt` complete
- [ ] `.gitignore` present
- [ ] Clear commit history (7+ meaningful commits)
- [ ] Repository description + topics set
- [ ] At least 1 dashboard screenshot in README

---

## ✅ Proof Checklist

### Technical Proof
- [ ] `data/raw/sensor_data.csv` exists (5000 rows)
- [ ] `data/processed/processed_data.csv` exists (20+ feature columns)
- [ ] `models/best_model.pkl` exists
- [ ] `models/scaler.pkl` exists
- [ ] `outputs/graphs/confusion_matrix.png`
- [ ] `outputs/graphs/roc_curves.png`
- [ ] `outputs/graphs/feature_importance.png`
- [ ] `outputs/graphs/failure_timeline.png`
- [ ] `outputs/graphs/sensor_trends.png`
- [ ] `outputs/reports/alert_log.csv`
- [ ] `logs/pipeline.log`

### GitHub Proof
- [ ] Repo is public
- [ ] 7+ commits with meaningful messages
- [ ] README has model results table
- [ ] README has architecture diagram
- [ ] README has installation instructions
- [ ] Topics/tags added to repo

### Demo Proof
- [ ] `python main.py` runs without errors
- [ ] `streamlit run dashboard/app.py` launches dashboard
- [ ] Both demo scenarios show different alert levels
- [ ] All graphs save to `outputs/graphs/`

---

## 📸 Screenshot Naming Convention

Save all screenshots as:
```
images/
├── 01_dataset_preview.png
├── 02_feature_engineering.png
├── 03_model_metrics.png
├── 04_confusion_matrix.png
├── 05_roc_curves.png
├── 06_feature_importance.png
├── 07_failure_timeline.png
├── 08_sensor_trends.png
├── 09_dashboard_normal.png
├── 10_dashboard_critical.png
├── 11_demo_prediction.png
└── 12_github_repo_preview.png
```

Add the best 3-4 screenshots to your README for maximum impact.
