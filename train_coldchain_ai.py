import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import random

# -------------------------------------------------------------------
# STEP 1: Generate Synthetic Dataset
# -------------------------------------------------------------------

def generate_dataset(produce_name, n_samples=1000):
    cities = ["Shimla", "Delhi", "Nagpur", "Hyderabad", "Bangalore", "Chennai"]
    data = []

    for _ in range(n_samples):
        city = random.choice(cities)
        latitude = round(random.uniform(8.0, 32.0), 6)
        longitude = round(random.uniform(70.0, 85.0), 6)
        outside_temp = random.uniform(5, 35)

        if produce_name == "Apple":
            base_set = 4.0
            humidity = random.uniform(85, 95)
        else:
            base_set = 13.0
            humidity = random.uniform(80, 90)

        # Smart setpoint base logic
        smart_setpoint = base_set - ((outside_temp - 25.0) / 10.0 if outside_temp > 25 else 0)
        smart_setpoint = max(smart_setpoint, base_set - 2)

        temperature = smart_setpoint + random.uniform(-2, 2)
        voc = random.randint(500, 3000)
        cooler = 1 if temperature > smart_setpoint else 0
        heater = 1 if temperature < smart_setpoint else 0
        alert = 1 if voc > 2500 else 0

        # AI target outputs
        optimal_setpoint = smart_setpoint - random.uniform(-0.5, 0.5)
        shelf_life = max(1, 100 - abs(temperature - optimal_setpoint)*10 - (voc/1000)*5)

        data.append([
            produce_name, temperature, humidity, voc, latitude, longitude, city,
            outside_temp, smart_setpoint, cooler, heater, alert,
            optimal_setpoint, shelf_life
        ])

    df = pd.DataFrame(data, columns=[
        "produce", "temperature", "humidity", "voc", "latitude", "longitude", "city",
        "outside_temp", "setpoint", "cooler", "heater", "alert",
        "optimal_setpoint", "predicted_shelf_life_days"
    ])
    return df


# Generate datasets for Apple and Banana
df_apple = generate_dataset("Apple", 1000)
df_banana = generate_dataset("Banana", 1000)

# Combine and shuffle
df = pd.concat([df_apple, df_banana]).sample(frac=1).reset_index(drop=True)

print("✅ Dataset created with shape:", df.shape)
print(df.head())

# -------------------------------------------------------------------
# STEP 2: Clean + Encode the Data
# -------------------------------------------------------------------

# Encode categorical columns
le_produce = LabelEncoder()
df["produce"] = le_produce.fit_transform(df["produce"])  # Apple=0, Banana=1

le_city = LabelEncoder()
df["city"] = le_city.fit_transform(df["city"])           # Encode city names

# Save encoders for Flask API usage later
with open("label_encoders.pkl", "wb") as f:
    pickle.dump({"produce": le_produce, "city": le_city}, f)

# -------------------------------------------------------------------
# STEP 3: Train AI Models
# -------------------------------------------------------------------

features = [
    "produce", "temperature", "humidity", "voc", "latitude", "longitude",
    "city", "outside_temp", "setpoint", "cooler", "heater", "alert"
]

# Train-Test Split
X = df[features]
y_setpoint = df["optimal_setpoint"]
y_shelf = df["predicted_shelf_life_days"]

X_train, X_test, y1_train, y1_test = train_test_split(X, y_setpoint, test_size=0.2, random_state=42)
X_train2, X_test2, y2_train, y2_test = train_test_split(X, y_shelf, test_size=0.2, random_state=42)

# Model 1 — Optimal Setpoint
model_setpoint = RandomForestRegressor(n_estimators=120, random_state=42)
model_setpoint.fit(X_train, y1_train)
pred1 = model_setpoint.predict(X_test)

# Model 2 — Shelf Life (days)
model_shelf = RandomForestRegressor(n_estimators=120, random_state=42)
model_shelf.fit(X_train2, y2_train)
pred2 = model_shelf.predict(X_test2)

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
print("\n📊 Model Performance:")
print(f"Optimal Setpoint MAE: {mean_absolute_error(y1_test, pred1):.3f}")
print(f"Optimal Setpoint R²: {r2_score(y1_test, pred1):.3f}")
print(f"Shelf Life MAE: {mean_absolute_error(y2_test, pred2):.3f}")
print(f"Shelf Life R²: {r2_score(y2_test, pred2):.3f}")

# -------------------------------------------------------------------
# Save Models
# -------------------------------------------------------------------
pickle.dump(model_setpoint, open("optimal_setpoint_model.pkl", "wb"))
pickle.dump(model_shelf, open("shelf_life_model.pkl", "wb"))

# Save dataset for inspection
df.to_csv("coldchain_dataset.csv", index=False)

print("\n✅ Models and dataset successfully saved!")
print("📁 Files created:")
print(" - coldchain_dataset.csv")
print(" - optimal_setpoint_model.pkl")
print(" - shelf_life_model.pkl")
print(" - label_encoders.pkl")
