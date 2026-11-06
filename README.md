# CLV Prediction & Cohort Retention Analysis

End-to-end analytics project on the **Olist Brazilian E-Commerce** dataset: a machine-learning pipeline that predicts whether a customer will place a repeat order and estimates their Customer Lifetime Value (CLV), plus a zero-dependency **Node.js dashboard** with a **live prediction page**.

---

## Features

- **ML pipeline (Jupyter Notebook)**: data cleaning, outlier detection, feature engineering, leakage checks, model comparison, tuning, and CLV scoring.
- **Repeat-purchase classifier**: Logistic Regression, Decision Tree, Naive Bayes, KNN, Random Forest, XGBoost and LightGBM compared; final model is a tuned LightGBM.
- **Spend regressor**: predicts expected spend (Linear/Ridge/Lasso/ElasticNet, Random Forest, XGBoost, LightGBM).
- **CLV score** = `P(repeat purchase) × predicted spend`, with customers segmented into **Platinum / Gold / Silver / Bronze**.
- **Cohort retention analysis** and business insights (revenue concentration, delivery vs. reviews).
- **Analytics dashboard**: summary KPIs, monthly trends, categories, states, payments, reviews, delivery, and orders explorer, with year/state filters.
- **Live Predictions page**: enter customer features in the browser and get a prediction instantly (LightGBM + XGBoost trees evaluated directly in Node.js, no extra packages).

---

## Key Results

| Item | Value |
|---|---|
| Orders / customers | 99,441 orders / 96,096 customers |
| Customers used for modelling | 57,143 (orders before 2018-03-01) |
| Repeat-purchase rate | 2.66% (≈ 36.6 : 1 class imbalance) |
| Top 10% customers | ≈ 35.9% of revenue |
| Late-delivery rate | 8.1% |
| Avg. review (late vs. on-time) | 2.62 vs. 4.30 |
| Final LightGBM | Val AUC 0.661 / Test AUC 0.582 |
| Tuned threshold (val F1) | 0.215 |

> **Note:** Repeat purchase is extremely rare in this dataset (~2.7%), so predictive power is modest. The pipeline is designed to avoid inflated scores: leakage-prone columns (`frequency`, `tenure_days`, `monetary`) are excluded from features, and SMOTE is applied **only on the training split**; validation and test sets remain untouched.

---

## Project Structure

```
olist-dashboard/
├── olist_clv_pipeline.ipynb   # Full ML pipeline (EDA -> models -> CLV segments)
├── server.js                  # Zero-dependency Node.js server + API routes
├── package.json
├── lib/
│   ├── data.js                # CSV loading & aggregation for the dashboard
│   └── predict.js             # Prediction engine (LightGBM + XGBoost in Node)
├── data/
│   ├── models.json            # Model comparison results
│   └── model_live.json        # Exported trained model for live predictions
├── public/
│   ├── index.html             # Dashboard
│   ├── predict.html           # Live predictions page
│   ├── css/
│   └── js/
├── ml/
│   └── train_model.py         # Retrain + export model for live predictions
├── README.md
└── README_LIVE.md             # Notes for the live-predictions add-on
```

> The dataset CSV is **not** included in the repo (see `.gitignore`). Place `olist_merged.csv` in the `data/` folder.

---

## Tech Stack

- **Python**: pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), LightGBM, XGBoost, statsmodels, scipy, matplotlib, seaborn, Optuna (optional)
- **Backend**: Node.js (>= 16), built-in modules only, no npm dependencies
- **Frontend**: HTML, CSS, vanilla JavaScript

---

## Getting Started

### 1. Run the dashboard

```bash
git clone <your-repo-url>
cd olist-dashboard
# put olist_merged.csv inside the data/ folder
node server.js
```

Open:
- Dashboard: http://localhost:3000
- Live predictions: http://localhost:3000/predict.html

The first start loads the CSV and can take 5–20 seconds.

### 2. Run the notebook

```bash
pip install pandas numpy scikit-learn imbalanced-learn lightgbm xgboost statsmodels scipy matplotlib seaborn optuna jupyter
jupyter notebook olist_clv_pipeline.ipynb
```

Set the dataset path in the config cell, then run all cells. Outputs (including `clv_scored_customers.csv`) are saved to `clv_outputs/`.

### 3. Retrain the live model (optional)

```bash
pip install pandas scikit-learn lightgbm xgboost imbalanced-learn
python ml/train_model.py --csv "path/to/olist_merged.csv"
```

Restart the server afterwards.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/options` | Filter options (years, states) |
| GET | `/api/summary` | KPI summary |
| GET | `/api/monthly` | Monthly trends |
| GET | `/api/categories` | Category stats |
| GET | `/api/states` | State-wise stats |
| GET | `/api/payments` | Payment methods |
| GET | `/api/reviews` | Review distribution |
| GET | `/api/delivery` | Delivery performance |
| GET | `/api/orders` | Orders explorer |
| GET | `/api/models` | Model comparison results |
| GET | `/api/predict/meta` | Form defaults + model metrics |
| GET | `/api/predict/sample` | Random held-out real customer with actual result |
| POST | `/api/predict` | Predict for a custom customer |

Filters (`year`, `state`) can be passed as query parameters on the summary endpoints.

**Example `POST /api/predict` body:**

```json
{
  "avg_order_value": 150,
  "avg_price": 120,
  "avg_freight_ratio": 0.2,
  "avg_delivery_days": 12,
  "late_rate": 0,
  "avg_review": 4.5,
  "n_categories": 1,
  "weekend_rate": 0.3,
  "recency_days": 200,
  "state": "SP",
  "payment": "credit_card"
}
```

---

## Methodology (Summary)

1. Clean and merge the Olist tables; detect outliers (Isolation Forest).
2. Build a customer table from an **observation window** (before the split date) and define targets from the later period.
3. Check for leakage and multicollinearity (VIF).
4. Split → scale → SMOTE (train only).
5. Compare multiple classifiers and regressors; tune the final LightGBM with early stopping.
6. Pick the decision threshold on validation F1; calibrate probabilities back to the true prior.
7. Compute CLV score and assign customer segments.

---

## Dataset

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle).

---

## License

For educational / academic use.
