# Customer Churn Prediction System

An end-to-end machine learning project that predicts whether a telecom customer is likely to leave the company.

## Problem Statement
Telecom companies lose revenue when customers cancel. This system identifies at-risk customers before they leave, so the company can take action.

## Technologies
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, SQL, Power BI, Streamlit, Jupyter Notebook

## Project Workflow
1. Data loading and cleaning (7043 customers, 21 columns)
2. Exploratory data analysis (EDA)
3. Statistical analysis (t-test, chi-square)
4. Feature engineering (4 new features created)
5. Train/test split (80/20, stratified)
6. Four ML models trained and compared
7. XGBoost selected and tuned with RandomizedSearchCV
8. SHAP explainability
9. SQL analysis (10 queries)
10. Power BI dashboard
11. Streamlit web application

## Dataset
IBM Telco Customer Churn dataset (Kaggle)
- 7,043 customers
- 21 columns
- 26.54% churn rate

## ML Models and Results

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.798 | 0.519 | 0.577 | 0.843 |
| Decision Tree | 0.798 | 0.567 | 0.599 | 0.830 |
| Random Forest | 0.803 | 0.500 | 0.575 | 0.843 |
| XGBoost (tuned) | 0.808 | 0.532 | 0.595 | 0.845 |

Best model: **XGBoost** (highest ROC-AUC, no overfitting)

## Key EDA Findings
- Month-to-month contracts: 42.7% churn vs 2.8% for two-year contracts
- First 12 months: 47.4% churn rate (danger zone)
- Electronic check payment: 45.3% churn
- No online security: 41.8% churn vs 14.6% with security
- Fiber optic internet: 41.9% churn

## SHAP Top Factors
1. Contract_Month-to-month (+0.64)
2. tenure (+0.44)
3. OnlineSecurity_No (+0.25)
4. MonthlyCharges (+0.24)
5. InternetService_Fiber optic (+0.22)

## Business Insights
- High-risk group (969 customers): 71.3% actual churn rate
- Medium-risk group (1697 customers): 44.1% actual churn rate
- Medium + High = 38% of customers but ~77% of all churners
- New + month-to-month + fiber optic: 916 customers, 70.2% churn

## How to Run

### Setup
```bash
git clone https://github.com/Ranjith067/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt