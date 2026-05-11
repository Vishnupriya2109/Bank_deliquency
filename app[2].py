"""
CRIS - Customer Risk Intelligence System
app.py — Flask REST API Backend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

# ──────────────────────────────────────────────
# LOAD MODEL BUNDLE
# ──────────────────────────────────────────────
bundle          = pickle.load(open("model.pkl", "rb"))
model           = bundle["model"]
feature_columns = bundle["feature_columns"]
month_map       = bundle["month_map"]

MONTH_MAP = {"On-time": 0, "Late": 1, "Missed": 2}

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def risk_category(prob: float) -> str:
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    return "Low Risk"

def risk_color(level: str) -> str:
    return {"High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#27ae60"}[level]

def build_feature_vector(data: dict) -> np.ndarray:
    """
    Build the feature vector in the exact order the model was trained on.
    Accepts a dict with all raw customer fields.
    """
    months = [MONTH_MAP.get(data.get(f"Month_{i}", "On-time"), 0) for i in range(1, 7)]
    payment_score = sum(months)
    recent_trend  = sum(months[3:]) - sum(months[:3])
    credit_util   = float(data.get("Credit_Utilization", 0.5))
    dti           = float(data.get("Debt_to_Income_Ratio", 0.3))
    missed        = float(data.get("Missed_Payments", 0))
    credit_score  = float(data.get("Credit_Score", 600))

    risk_composite = (
        credit_util * 0.30 +
        dti         * 0.30 +
        (missed / 6) * 0.25 +
        (1 - credit_score / 850) * 0.15
    )

    emp_status   = data.get("Employment_Status", "employed").strip().lower().replace("self-employed", "self_employed").replace("emp", "employed")
    card_type    = data.get("Credit_Card_Type", "Standard")
    location     = data.get("Location", "Chicago")

    row = {col: 0 for col in feature_columns}

    # Numeric fields
    row["Age"]                  = float(data.get("Age", 35))
    row["Income"]               = float(data.get("Income", 60000))
    row["Credit_Score"]         = credit_score
    row["Credit_Utilization"]   = credit_util
    row["Missed_Payments"]      = missed
    row["Loan_Balance"]         = float(data.get("Loan_Balance", 10000))
    row["Debt_to_Income_Ratio"] = dti
    row["Account_Tenure"]       = float(data.get("Account_Tenure", 5))

    for i, v in enumerate(months, 1):
        row[f"Month_{i}"] = v

    row["payment_pattern_score"] = payment_score
    row["recent_payment_trend"]  = recent_trend
    row["risk_composite"]        = risk_composite

    # One-hot
    emp_key = f"Employment_Status_{emp_status}"
    if emp_key in row:
        row[emp_key] = 1

    card_key = f"Credit_Card_Type_{card_type}"
    if card_key in row:
        row[card_key] = 1

    loc_key = f"Location_{location}"
    if loc_key in row:
        row[loc_key] = 1

    return np.array([row[col] for col in feature_columns]).reshape(1, -1)


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"status": "CRIS API running", "version": "2.0"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Support both full-form dict and legacy flat features array
        if "features" in data and isinstance(data["features"], list):
            features = np.array(data["features"]).reshape(1, -1)
        else:
            features = build_feature_vector(data)

        prob = float(model.predict_proba(features)[0][1])
        pred = int(model.predict(features)[0])
        level = risk_category(prob)

        return jsonify({
            "prediction":  pred,
            "risk_score":  round(prob, 4),
            "risk_level":  level,
            "risk_color":  risk_color(level),
            "risk_percent": round(prob * 100, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "features": len(feature_columns)})


if __name__ == "__main__":
    print("🚀 CRIS API starting on http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
