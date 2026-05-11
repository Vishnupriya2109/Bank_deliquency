"""
CRIS - Customer Risk Intelligence System
train.py — Model Training Pipeline
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
df = pd.read_excel("Delinquency_prediction_dataset.xlsx")
df_original = df.copy()
print(f" Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ──────────────────────────────────────────────
# 2. CLEAN & PREPROCESS
# ──────────────────────────────────────────────
df['Employment_Status'] = (
    df['Employment_Status']
    .str.strip().str.lower()
    .replace({'emp': 'employed', 'self-employed': 'self_employed'})
)

df.fillna(df.median(numeric_only=True), inplace=True)

month_map = {"On-time": 0, "Late": 1, "Missed": 2}
month_cols = ["Month_1", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6"]
for col in month_cols:
    df[col] = df[col].map(month_map)

# ──────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────
df['payment_pattern_score'] = df[month_cols].sum(axis=1)
df['recent_payment_trend'] = (
    df[['Month_4', 'Month_5', 'Month_6']].sum(axis=1)
    - df[['Month_1', 'Month_2', 'Month_3']].sum(axis=1)
)
df['risk_composite'] = (
    df['Credit_Utilization']     * 0.30 +
    df['Debt_to_Income_Ratio']   * 0.30 +
    (df['Missed_Payments'] / 6)  * 0.25 +
    (1 - df['Credit_Score'] / 850) * 0.15
)

# One-hot encode categoricals
df = pd.get_dummies(
    df,
    columns=['Employment_Status', 'Credit_Card_Type', 'Location'],
    drop_first=False
)

# ──────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
X = df.drop(["Delinquent_Account", "Customer_ID"], axis=1)
y = df["Delinquent_Account"]
feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ──────────────────────────────────────────────
# 5. TRAIN MODEL
# ──────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
print(" Model trained")

# ──────────────────────────────────────────────
# 6. EVALUATE
# ──────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"   AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
fi = (
    pd.Series(model.feature_importances_, index=feature_columns)
    .sort_values(ascending=False)
)
print("\n Top 10 Feature Importances:")
print(fi.head(10).to_string())

# ──────────────────────────────────────────────
# 7. RISK SCORING ON TEST SET
# ──────────────────────────────────────────────
def risk_category(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

customer_data = df_original.iloc[y_test.index].copy()
customer_data["Predicted_Delinquent"] = y_pred
customer_data["Risk_Score"]           = y_prob
customer_data["Risk_Level"]           = [risk_category(p) for p in y_prob]
customer_data = customer_data.sort_values("Risk_Score", ascending=False)

print("\n Top 10 High-Risk Customers:")
print(customer_data[["Customer_ID", "Credit_Score", "Missed_Payments",
                      "Risk_Score", "Risk_Level"]].head(10).to_string(index=False))

customer_data.to_excel("customer_risk_output.xlsx", index=False)
print("\n Output saved: customer_risk_output.xlsx")

# ──────────────────────────────────────────────
# 8. SAVE MODEL BUNDLE
# ──────────────────────────────────────────────
model_bundle = {
    'model':           model,
    'feature_columns': feature_columns,
    'month_map':       month_map
}
pickle.dump(model_bundle, open("model.pkl", "wb"))
print("Model bundle saved: model.pkl")
