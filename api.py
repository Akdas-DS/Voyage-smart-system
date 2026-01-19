from flask import Flask, request, jsonify
import pandas as pd
import joblib

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load trained flight model
# -----------------------------
flight_model = joblib.load("models/flight_price_model.pkl")
flight_columns = joblib.load("models/flight_columns.pkl")

# -----------------------------
# Health check endpoint
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Flight Price API is running"}), 200

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    required_fields = ["distance", "time", "agency", "flightType", "from", "to"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    input_df = pd.DataFrame([{
        "distance": data["distance"],
        "time": data["time"],
        "agency": data["agency"],
        "flightType": data["flightType"],
        "from": data["from"],
        "to": data["to"]
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=flight_columns, fill_value=0)

    prediction = flight_model.predict(input_encoded)[0]

    return jsonify({
        "predicted_flight_price": round(float(prediction), 2)
    })

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
