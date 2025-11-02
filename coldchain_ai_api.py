from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# --------------------------------------------------------------
# Load models and feature order
# --------------------------------------------------------------
try:
    setpoint_model = pickle.load(open("optimal_setpoint_model.pkl", "rb"))
    shelf_model = pickle.load(open("shelf_life_model.pkl", "rb"))
    encoders = pickle.load(open("label_encoders.pkl", "rb"))
    print("✅ Models and encoders loaded successfully.")
except Exception as e:
    print("❌ Model load failed:", e)

# Define expected feature order (must match training)
expected_features = [
    "produce", "temperature", "humidity", "voc",
    "latitude", "longitude", "city",
    "outside_temp", "setpoint", "cooler",
    "heater", "alert"
]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"📩 Incoming keys: {list(data.keys())}")

        # Validate all fields
        missing = [f for f in expected_features if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Convert to DataFrame in correct column order
        df = pd.DataFrame([[data[f] for f in expected_features]],
                          columns=expected_features)

        # Encode categorical fields
        df["produce"] = encoders["produce"].transform(df["produce"])
        df["city"] = encoders["city"].transform(df["city"])

        # Predict
        optimal_temp = float(setpoint_model.predict(df)[0])
        shelf_days = float(shelf_model.predict(df)[0])

        return jsonify({
            "optimal_setpoint": round(optimal_temp, 2),
            "predicted_shelf_life_days": round(shelf_days, 1)
        })

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
