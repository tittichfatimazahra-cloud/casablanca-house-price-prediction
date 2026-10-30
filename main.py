
import pandas as pd
import matplotlib.pyplot as plt
import os

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Get the folder containing main.py
BASE_DIR = Path(__file__).resolve().parent

# Locate the dataset
casa_file_path = BASE_DIR / "housse_price_casablanca.csv"

# Read the dataset
casa_data = pd.read_csv(casa_file_path)

# 2. Select features and target
features = [
    "chambres",
    "salles_de_bain",
    "surface_m2",
    "age_bien_ans",
    "garage"
]

X = casa_data[features]
y = casa_data["price"]

# 3. Split the data
train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create a baseline model
baseline_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DummyRegressor(strategy="mean"))
])

baseline_model.fit(train_X, train_y)

baseline_predictions = baseline_model.predict(val_X)

baseline_mae = mean_absolute_error(
    val_y, baseline_predictions
)

# 5. Create Random Forest model
casa_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

# 6. Train model
casa_model.fit(train_X, train_y)

# 7. Predict house prices
predictions = casa_model.predict(val_X)

# 8. Evaluate model
mae = mean_absolute_error(val_y, predictions)

rmse = mean_squared_error(
    val_y, predictions
) ** 0.5

r2 = r2_score(val_y, predictions)

print("CASABLANCA HOUSE PRICE PREDICTION")
print("--------------------------------")
print(f"Baseline MAE: {baseline_mae:,.2f}")
print(f"Random Forest MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R² Score: {r2:.3f}")

# 9. Create images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# 10. Actual vs predicted visualization
plt.figure(figsize=(8, 6))

plt.scatter(val_y, predictions, alpha=0.4)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Casablanca: Actual vs Predicted Prices")

plt.tight_layout()
plt.savefig("images/actual_vs_predicted.png")
plt.show()

# 11. Feature importance
rf_casa_model = casa_model.named_steps["model"]

importance = pd.Series(
    rf_casa_model.feature_importances_,
    index=features
).sort_values()

importance.plot(kind="barh", figsize=(8, 5))

plt.title("Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.show()

# 12. Save predictions
results = pd.DataFrame({
    "Actual Price": val_y.values,
    "Predicted Price": predictions
})

results.to_csv("predictions.csv", index=False)

print("\nPredictions saved successfully!")
