from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "flight_price_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "..", "models", "flight_columns.pkl")

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Voyage Analytics Flight Price API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(input_df)[0]
        return jsonify({"predicted_price": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
