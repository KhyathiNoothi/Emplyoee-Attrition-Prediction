from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

# Flask App
app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)

# Load saved files
model = joblib.load("xgb_attrition_model.pkl")
model_columns = joblib.load("model_columns.pkl")
threshold = joblib.load("decision_threshold.pkl")
nominal_cols = joblib.load("nominal_cols.pkl")

# Business Travel encoding
travel_map = {
    "Non-Travel": 0,
    "Travel_Rarely": 1,
    "Travel_Frequently": 2
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from frontend
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Binary Encoding
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({
                "Male": 1,
                "Female": 0
            })

        if "OverTime" in df.columns:
            df["OverTime"] = df["OverTime"].map({
                "Yes": 1,
                "No": 0
            })

        # Business Travel Encoding
        if "BusinessTravel" in df.columns:
            df["BusinessTravel"] = df["BusinessTravel"].map(travel_map)

        # One-Hot Encoding
        df = pd.get_dummies(df, columns=nominal_cols)

        # Match training columns
        df = df.reindex(columns=model_columns, fill_value=0)

        # Convert all values to float
        df = df.astype(float)

        # Predict probability
        probability = model.predict_proba(df)[0][1]

        # Apply threshold
        prediction = int(probability >= threshold)

        return jsonify({
            "prediction": prediction,
            "probability": round(float(probability), 4),
            "label": "Likely to Leave" if prediction == 1 else "Likely to Stay"
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)