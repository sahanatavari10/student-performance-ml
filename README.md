# Student Performance ML

Predicting at-risk students using the [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) dataset (Mathematics course). A student is classified as **at-risk** when their final grade (G3) falls below 10 out of 20.

## Dataset

- **Source**: UCI Machine Learning Repository
- **Records**: 395 students
- **Features**: 30 demographic, social, and school-related attributes
- **Target**: Binary — `at_risk` (G3 < 10)

> **Note**: Period grades G1 and G2 are dropped to avoid data leakage.

## Project Structure

```
student-performance-ml/
├── data/                  # Raw and engineered datasets
├── models/                # Saved model artifacts
├── notebooks/
│   ├── 01_baseline_model.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_tuning.ipynb
│   └── 06_interpretability.ipynb
├── src/                   # Reusable Python modules
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

Run in order:

1. **01_baseline_model** — Original baseline (Logistic Regression, 89.87% accuracy)
2. **02_eda** — Exploratory data analysis: distributions, correlations, bivariate plots
3. **03_feature_engineering** — Drops G1/G2, creates `parent_edu`, `alcohol`, `studytime_failures`
4. **04_model_comparison** — 5 models + SMOTE, stratified 5-fold CV, confusion matrices, ROC curves
5. **05_tuning** — Hyperparameter tuning (RandomizedSearchCV), saves best model
6. **06_interpretability** — Feature importance & SHAP analysis

## Results

_Results will be updated after running the notebooks._
