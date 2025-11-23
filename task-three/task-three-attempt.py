import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Features and target
X = df[['credit_lines_outstanding', 'loan_amt_outstanding', 
        'total_debt_outstanding', 'income', 'years_employed', 'fico_score']]
y = df['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
logreg_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]
logreg_auc = roc_auc_score(y_test, logreg_pred_proba)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
rf_pred_proba = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred_proba)

print(f"Logistic Regression AUC: {logreg_auc:.3f}")
print(f"Random Forest AUC: {rf_auc:.3f}")

# Select best model (for simplicity, use Random Forest here)
best_model = rf
best_scaler = None  # Random Forest does not need scaling

# Expected Loss function
def expected_loss(loan_features, model=best_model, scaler=best_scaler, recovery_rate=0.1):
    """
    loan_features: dict with keys
        'credit_lines_outstanding', 'loan_amt_outstanding',
        'total_debt_outstanding', 'income', 'years_employed', 'fico_score'
    """
    import numpy as np
    X_input = pd.DataFrame([loan_features])
    
    if scaler:
        X_input_scaled = scaler.transform(X_input)
        pd_pred = model.predict_proba(X_input_scaled)[:, 1][0]
    else:
        pd_pred = model.predict_proba(X_input)[:, 1][0]
    
    EL = pd_pred * loan_features['loan_amt_outstanding'] * (1 - recovery_rate)
    return EL, pd_pred

# Example usage
sample_loan = {
    'credit_lines_outstanding': 2,
    'loan_amt_outstanding': 5000,
    'total_debt_outstanding': 15000,
    'income': 60000,
    'years_employed': 5,
    'fico_score': 650
}

el, pd_pred = expected_loss(sample_loan)
print(f"Predicted PD: {pd_pred:.3f}, Expected Loss: {el:.2f}")
