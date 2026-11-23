
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "city_day.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

print("Dataset loaded:", df.shape)
print("Columns:", df.columns.tolist())

# Define target and pollutant columns
target = "AQI"

pollutant_columns = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

if target not in df.columns or "City" not in df.columns:
    raise ValueError("Dataset must contain City and AQI columns.")

# Keep pollutant columns that contain at least one numeric reading
available_features = []

for col in pollutant_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().any():
            available_features.append(col)

if not available_features:
    raise ValueError("No usable pollutant columns found.")

# Clean target
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target, "City"]).copy()

# Prepare features
df["City"] = df["City"].fillna("Unknown").astype(str)

X = df[["City"] + available_features].copy()
y = df[target].copy()

print("\nTarget: AQI")
print("Features:", X.columns.tolist())
print("Usable records:", len(X))

# Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, available_features),
    ("categorical", categorical_transformer, ["City"])
])

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# Evaluate
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save model and metadata
bundle = {
    "model": pipeline,
    "features": available_features,
    "cities": sorted(df["City"].unique().tolist())
}

joblib.dump(bundle, MODEL_DIR / "aqi_model.pkl")

metrics = {
    "algorithm": "Random Forest Regressor",
    "target": "AQI",
    "MAE": round(float(mae), 2),
    "RMSE": round(float(rmse), 2),
    "R2 Score": round(float(r2), 4),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved in models/aqi_model.pkl")
print("Metrics saved in outputs/metrics.json")
print("Training completed successfully!")
