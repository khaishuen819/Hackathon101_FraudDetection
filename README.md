# Car Insurance Fraud Detection

Logistic regression model that flags potentially fraudulent car insurance claims, plus an analysis of which claim segments carry the highest fraud risk.

Built in 2 days at **Hackathon Workshop 101** (APU Actuarial Science Club, 17–18 January 2026). The team placed **2nd Runner-Up out of 9 teams**.

---

## Problem

Insurers lose money when fraudulent claims slip through. Investigating every claim, though, is expensive.

This project answers two research questions:

1. **RQ1 — Where is fraud concentrated?** Which policyholder, vehicle, fault and timing segments have the highest fraud rates?
2. **RQ2 — Can we score claims by fraud risk?** Build a model that catches as many fraudulent claims as possible, so investigators know which ones to review first.

## Dataset

- `carclaims.csv`, provided by the hackathon organisers
- **15,420 claims**, 33 columns: policyholder, vehicle, policy, accident and claim details
- Target: `FraudFound` (Yes / No)
- **Severe class imbalance:** 923 fraudulent claims (**5.99%**) vs 14,497 genuine (94.01%)

> The dataset is not included in this repository. To run the notebook, place `[Dataset] carclaims.csv` in the same folder.

## Approach

1. **Cleaning:** stripped whitespace from column names and categorical values; encoded `FraudFound` as 0/1.
2. **Segment analysis (RQ1):** calculated and charted the fraud rate across 7 dimensions:
   - Fault
   - Policyholder age
   - Vehicle category
   - Accident area
   - Days from policy start to accident
   - Day of the week of the accident
   - Day of the week the claim was filed
3. **Feature engineering:**
   - 8 categorical features, one-hot encoded with `drop_first=True` to avoid multicollinearity.
   - 3 numeric features: age, deductible, driver rating.
   - 22 model features in total.
4. **Train/test split:** 70/30, **stratified** so both sets keep the 94:6 fraud ratio (test set: 4,626 claims).
5. **Model (RQ2):** logistic regression with `class_weight='balanced'`, so the rare fraud class is not ignored.
6. **Evaluation:** ROC-AUC, recall, precision and the confusion matrix. Accuracy is not used as the headline metric, because a model that labels every claim "genuine" would score 94% accuracy while catching no fraud.
7. **Output:** a fraud risk score (predicted probability) for each test-set policy, exported to `fraud_risk_scores.csv`.

## Results (test set, 4,626 claims)

| Metric | Value | What it means |
|---|---|---|
| ROC-AUC | **0.744** | How well the model ranks fraudulent claims above genuine ones |
| Recall | **88.1%** | Caught 244 of 277 fraudulent claims; missed 33 |
| Precision | 11.9% | About 1 in 8 flagged claims is actually fraud |
| Claims flagged for review | 44.2% | 2,044 of 4,626 |

**Trade-off:** the model is tuned to *catch fraud*, not to minimise false alarms. It misses very few fraudulent claims, at the cost of 1,800 genuine claims being flagged for review. That suits a first-pass screening tool, where a missed fraud costs more than an extra check.

## Key findings

**Fraud rate by segment**

| Segment | Fraud rate |
|---|---|
| Policyholder at fault | **7.89%** |
| Third party at fault | 0.88% |
| Utility vehicles | **11.25%** |
| Sedans | 8.22% |
| Sports cars | 1.57% |
| Rural accidents | **8.32%** |
| Urban accidents | 5.72% |
| Policyholders aged 21–25 | **14.8%** (small group: 108 claims) |

- Claims where the policyholder was at fault have roughly **9× the fraud rate** of third-party-fault claims.

**Strongest model predictors** (logistic regression coefficients)
- **Lower fraud risk:**
  - Third party at fault (−2.55)
  - Sports vehicle (−1.99)
  - Policy sold by an internal agent (−1.94)
- **Higher fraud risk:**
  - Policyholder aged 18–20 (+1.53)
  - Policyholder aged 21–25 (+0.98)

## Limitations and next steps

- **Default threshold:** the model uses the default 0.5 classification threshold. Tuning it with a precision–recall curve could cut false positives.
- **Single split:** no cross-validation or hyperparameter tuning (2-day hackathon scope).
- **Other models:** tree-based models (Random Forest, XGBoost) may capture interactions that logistic regression misses.
- **Small segments:** some segments, such as policyholders aged 18–20, are very small, so treat their fraud rates with caution.

## Tech stack

- Python 3.10
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tabulate jupyter
jupyter notebook APU_Hackathon_Fraud_Detection.ipynb
```

`tabulate` is needed for the `to_markdown()` tables printed in the notebook.

## Repository structure

```
├── APU_Hackathon_Fraud_Detection.ipynb   # Full analysis and model
└── README.md
```

## Team

Built by a team of 5 at Hackathon Workshop 101.

- **Model code:** Chew Khai Shuen ([GitHub](https://github.com/khaishuen819)) and [Teammate name], with Gemini as a coding assistant
- **[Other roles, e.g. business analysis and presentation]:** [Teammate names]
