
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# 1. Load Indian housing data
df = pd.read_csv(
    r"C:\Users\ASUS\OneDrive\Desktop\House Price prediction\Bengaluru_House_Data.csv"
)

df.columns = df.columns.str.strip()

required = ["location", "size", "total_sqft", "bath", "price"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 2. Clean square-foot values, including ranges like "1200-1500"
def convert_sqft(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    numbers = re.findall(r"\d+\.?\d*", value)

    if not numbers:
        return np.nan

    if "-" in value and len(numbers) >= 2:
        return (float(numbers[0]) + float(numbers[1])) / 2

    return float(numbers[0])

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# 3. Extract BHK count from values such as "2 BHK" or "4 Bedroom"
df["bhk"] = pd.to_numeric(
    df["size"].astype(str).str.extract(r"(\d+)")[0],
    errors="coerce"
)

# 4. Clean numerical fields
for col in ["bath", "balcony", "price"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Missing balconies are filled later by the pipeline
if "balcony" not in df.columns:
    df["balcony"] = np.nan

# Remove rows without essential information
df = df.dropna(
    subset=["location", "total_sqft", "bhk", "bath", "price"]
)

# Remove obviously invalid or extreme records for this demo
df = df[
    (df["total_sqft"] > 100)
    & (df["total_sqft"] < 20000)
    & (df["bhk"] >= 1)
    & (df["bhk"] <= 20)
    & (df["bath"] >= 1)
    & (df["price"] > 0)
    & (df["price"] < 2000)
].copy()

# 5. Use common Indian real-estate terminology
if "area_type" not in df.columns:
    df["area_type"] = "Not specified"

features = [
    "location", "area_type", "bhk",
    "total_sqft", "bath", "balcony"
]

X = df[features].copy()
y = df["price"]  # Price in lakh rupees

# Standardize location and area type text
X["location"] = X["location"].astype(str).str.strip()
X["area_type"] = X["area_type"].astype(str).str.strip()

# 6. Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = ["bhk", "total_sqft", "bath", "balcony"]
categorical_features = ["location", "area_type"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 7. Define models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained = {}

# 8. Train and compare
for name, estimator in models.items():
    print(f"Training {name}...")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", estimator)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE (₹ Lakhs)": mean_absolute_error(y_test, predictions),
        "RMSE (₹ Lakhs)": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2 Score": r2_score(y_test, predictions)
    })

    trained[name] = model

results_df = pd.DataFrame(results).sort_values(
    "RMSE (₹ Lakhs)"
)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

results_df.to_csv("model_results.csv", index=False)

# 9. Save the best model and app options
best_name = results_df.iloc[0]["Model"]

artifact = {
    "model": trained[best_name],
    "model_name": best_name,
    "features": features,
    "locations": sorted(X["location"].unique().tolist()),
    "area_types": sorted(X["area_type"].unique().tolist())
}

joblib.dump(artifact, "house_price_model.joblib")

print(f"\nBest model by test RMSE: {best_name}")
print("Saved model: house_price_model.joblib")
print("Prices are measured in lakh rupees.")
