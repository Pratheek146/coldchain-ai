# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# --- Load models (update paths as needed) ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
setpoint_model_path = os.path.join(MODEL_DIR, "optimal_setpoint_model.pkl")
shelf_model_path = os.path.join(MODEL_DIR, "shelf_life_model.pkl")

# load safely with error handling
try:
    with open(setpoint_model_path, "rb") as f:
        setpoint_model = pickle.load(f)
    with open(shelf_model_path, "rb") as f:
        shelf_model = pickle.load(f)
    MODELS_LOADED = True
except Exception as e:
    print("Model load error:", e)
    setpoint_model = None
    shelf_model = None
    MODELS_LOADED = False

# expected feature order used when training your model
EXPECTED_FEATURES = [
    "produce", "temperature", "humidity", "voc", "latitude",
    "longitude", "outside_temp", "setpoint", "cooler", "heater", "alert", "city"
]
# Note: adjust EXPECTED_FEATURES to exactly match the columns used during training.

@app.route("/")
def health():
    return jsonify({"status": "ok", "models_loaded": MODELS_LOADED})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    # Validate required fields
    missing = [k for k in EXPECTED_FEATURES if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Build the feature array in the same order as training
    try:
        # Convert categorical produce to numeric if necessary:
        produce = data["produce"]
        # if produce sent as "Apple"/"Banana", map to numeric: Apple->0, Banana->1
        if isinstance(produce, str):
            produce_map = {"Apple": 0, "Banana": 1}
            produce = produce_map.get(produce, 0)

        # If city is string, you must encode it the same way used in training.
        # If you trained using city_encoded, accept a numeric city instead.
        # Here we assume city is numeric encoded already OR training used city one-hot/encoded.
        city = data["city"]
        if isinstance(city, str):
            # fallback: use a simple static mapping (must match training mapping)
            city_map = {"Shimla":0,"Delhi":1,"Nagpur":2,"Hyderabad":3,"Bangalore":4,"Chennai":5}
            city = city_map.get(city, 0)

        X = [
            float(produce),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["voc"]),
            float(data["latitude"]),
            float(data["longitude"]),
            float(data["outside_temp"]),
            float(data["setpoint"]),
            int(data["cooler"]),
            int(data["heater"]),
            int(data["alert"]),
            float(city)
        ]

        X_arr = np.array(X).reshape(1, -1)

        # Try AI model predictions, fallback as needed
        fallback_used = False
        try:
            opt_setpoint = float(setpoint_model.predict(X_arr)[0])
        except Exception as e:
            print("Setpoint API model error:", e)
            # fallback to input setpoint (smart setpoint from device)
            opt_setpoint = float(data["setpoint"])
            fallback_used = True

        try:
            pred_shelf = float(shelf_model.predict(X_arr)[0])
        except Exception as e:
            print("Shelf life API model error:", e)
            pred_shelf = None
            fallback_used = True

        resp = {
            "optimal_setpoint": round(opt_setpoint, 2),
            "predicted_shelf_life_days": None if pred_shelf is None else round(pred_shelf, 2),
            "fallback_used": fallback_used
        }
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local debugging - but Spaces will use gunicorn via Dockerfile.
    app.run(host="0.0.0.0", port=5000)
