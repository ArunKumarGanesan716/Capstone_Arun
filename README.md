# Capstone Project — Investor Risk Profiling & Product Recommendation (Arun, UCB 2025)

Project notebook: `Capstone_Arun_UCB_2025.ipynb`

## Executive summary / Business objective

This project demonstrates how data and machine learning can support wealth-management decisions by: profiling investor risk, recommending primary investment products, detecting churn intent from client comments, and producing explainable, fairness-aware model outputs suitable for governance and advisor workflows.

Primary business objectives:
- Predict client risk tolerance to inform advisory nudges and suitability checks.
- Recommend a client's primary investment avenue (Equity, Bonds, Mutual Funds, Crypto, etc.) to drive cross-sell and personalization.
- Detect churn intent using sentiment + allocation mismatch to trigger retention actions.
- Produce an interpretable scorecard (performance + fairness + statistical tests) for governance and executive review.


## Dataset (exact values from notebook)

- File: `Capstone_Investors_100k.csv`
- Rows: 100,000
- Columns: 22
- Key fields: client_id, age, gender, education, city_tier, race, income, risk_tolerance, investment_objective, allocation percentages (equity_pct, bonds_pct, mutual_funds_pct, gold_pct, crypto_pct, etc.), mostly_invest_in, client_comment


## Hypotheses (tested)

- H1a: Income differs significantly across risk tolerance levels (ANOVA).
- H1b: Risk tolerance is associated with primary investment avenue (χ² test).
- H1c: Equity allocation % is higher for high-risk-tolerance clients.
- H1d: Negative sentiment correlates with higher churn intent (Spearman / correlation tests).


## Methods & workflow

The notebook implements a full ML lifecycle:
- Data loading, cleaning, missingness and outlier handling, and a cleaned CSV export.
- Feature engineering (Age_Bucket, Income_Bucket), sentiment extraction (VADER or lightweight lexicon), and ID-like column removal to avoid leakage.
- Exploratory data analysis (matplotlib, Plotly interactive charts, PCA, optional UMAP).
- Fast prototype modeling pipeline (dual preprocessors: sparse OHE for linear models, dense ordinal for tree-based models).
- Model training: classification, regression, and clustering with timing benchmarks and permutation importance.
- Fairness slicing and explainability via permutation importance and optional SHAP for tree models.


## Concrete outcomes and results (from notebook)

Dataset-level insights:
- Investors: predominantly young-to-middle-aged (mean age ~39).
- Income: mean ≈ $52,810; somewhat long-tailed distribution.
- Portfolios: equity-heavy (mean equity ~31%), small crypto exposure (~2.5%).

Modeling highlights (values taken from the notebook run):

- Classification (auto-picked categorical target; sample = 10,000):
   - Models tested (representative): HistGradientBoosting (HistGradBoost), LogisticRegression (LogReg), RandomForest, DecisionTree, GaussianNB, SVC-RBF, KNN.
   - Representative metrics:
      - HistGradBoost → Accuracy = 0.531, Macro-F1 = 0.175
      - LogReg        → Accuracy = 0.527, Macro-F1 = 0.188
   - Takeaway: classification tasks (e.g., predicting gender or small-cardinality categories) struggled due to class imbalance and weak signal for certain targets. Fairness concerns (bias toward majority class) were observed.

- Regression (target = income; sample = 10,000):
   - Models tested: HistGradientBoostingReg, RandomForestReg, LinearRegression, Lasso, Ridge, DecisionTreeReg.
   - Representative metrics (best performers):
      - HistGradientBoostReg → RMSE = 2.68, MAE = 2.18, R² = 0.942 (fit time ≈ 0.45s)
      - RandomForestReg      → RMSE = 2.70, MAE = 2.18, R² = 0.941
   - Takeaway: income is highly predictable (R² ≈ 0.94) and tree-based models performed best.

- Product recommendation & churn (specialized pipelines):
   - Product recommendation (target = `mostly_invest_in`) using HistGradientBoostingClassifier:
      - Accuracy ≈ 0.992, Macro-F1 ≈ 0.888 (noted in the notebook's scorecard)
   - Churn detection (derived `churn_intent` from sentiment/allocation mismatch):
      - Accuracy = 1.000, Macro-F1 = 1.000 (this reflects the engineered signal and should be validated on held-out / production data to avoid leakage)

- Hypothesis test H₁b (Risk tolerance vs Investment Avenue):
   - χ² = 61672.897, p < 0.0001 → reject H₀: strong association between risk tolerance and chosen investment avenue.


## Feature importance & top features

Permutation importance and SHAP (where applicable) highlighted different drivers for classification vs regression:

- Classification (top features observed in permutation importance / deployment report):
   - `client_comment` (text-derived signal)
   - `gold_pct`, `equity_pct`, `restriction_list`, `Income_Bucket`, `mutual_funds_pct`

- Regression (income prediction — top drivers):
   - `Age_Bucket` (dominant predictor)
   - Education level, bonds_pct, income bucket and several allocation percentages showed smaller but notable impacts (as per SHAP & PI).

Note: exact feature name encodings in the pipelines may appear as `num__<name>` or `cat__<name>` when OneHotEncoder / ColumnTransformer are used.


## Business impact & recommended actions

- Use the strong regression signal for income to power richer profiling and personalization (credit/risk-adjusted recommendations, advisor playbooks).
- Product recommendation baseline is promising for cross-sell; validate on recent production data and run fairness audits.
- Revisit classification tasks with: better-target engineering, rebalancing or class-weighting, and fairness-aware training if the objective requires classification (e.g., detecting specific cohorts).
- Treat perfect/near-perfect churn numbers with caution: review label creation to ensure no leakage and validate with a real holdout or prospective test.


## How to run (PowerShell)

Prerequisites:
- Python 3.8+ (virtualenv or conda recommended)
- Install packages from `requirements.txt` (created alongside this README)

Example (PowerShell):

```powershell
# create and activate a virtual environment (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# launch the notebook
jupyter notebook "Capstone_Arun_UCB_2025.ipynb"
```

If you encounter missing optional packages (e.g., `shap` or `umap-learn`), install them individually as needed.


## Files in this folder

- `Capstone_Arun_UCB_2025.ipynb` — main interactive notebook with data, models, tests and visuals.
- `README.md` — this file.
- `requirements.txt` — packages required to run the notebook.
- `Capstone_Investors_Cleaned.csv` — (generated by notebook) cleaned dataset used for modeling.


## Reproducibility notes

- The notebook sets a global random seed (RANDOM_STATE = 42) to make sampling and model runs reproducible.
- For large steps (UMAP, SHAP, heavy models), optional flags and sample-size caps are used to keep runtime feasible on a laptop.


Contact / Author

Author: Arun
Project: UCB Capstone — 2025

