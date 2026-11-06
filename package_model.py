
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# 1. Load modelling data and final classification configuration
df = pd.read_csv("data/modelling_table.csv")
config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]


X = df[FEATURES]
y = df["is_delayed"]


# 2. Recreate the same stratified train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# 3. Read the selected model settings
selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

depth_value = config["max_depth"]

if pd.isna(depth_value) or str(depth_value).strip().lower() in ["none", "nan", ""]:
    max_depth = None
else:
    max_depth = int(float(depth_value))


# 4. Build the final model pipeline
final_pipeline = Pipeline([
    ("selector", SelectKBest(f_classif, k=selector_k)),
    ("model", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    ))
])


# 5. Train the final model using training data
final_pipeline.fit(X_train, y_train)


# 6. Save the trained model
joblib.dump(final_pipeline, "models/delay_model.pkl")


# 7. Save the selected delay threshold
pd.DataFrame({
    "threshold": [threshold]
}).to_csv("models/delay_threshold.csv", index=False)


# 8. Save feature order
pd.DataFrame({
    "feature": FEATURES
}).to_csv("models/feature_order.csv", index=False)


# 9. Save the imputation values
raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)

pd.DataFrame([
    {"column": key, "median": value}
    for key, value in medians.items()
]).to_csv("models/impute_values.csv", index=False)


# 10. Reload the saved model and verify it
loaded_model = joblib.load("models/delay_model.pkl")

original_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
reloaded_probabilities = loaded_model.predict_proba(X_test)[:, 1]

max_difference = np.max(
    np.abs(original_probabilities - reloaded_probabilities)
)


# 11. Save reload verification results
pd.DataFrame({
    "test_rows_scored": [len(X_test)],
    "reload_max_abs_diff": [max_difference],
    "threshold": [threshold]
}).round(4).to_csv(
    "outputs/part_d_reload_check.csv",
    index=False
)


print("PACKAGE MODEL COMPLETED SUCCESSFULLY!")
print("Model saved: models/delay_model.pkl")
print("Threshold saved: models/delay_threshold.csv")
print("Feature order saved: models/feature_order.csv")
print("Imputation values saved: models/impute_values.csv")
print("Test rows verified:", len(X_test))
print("Reload difference:", round(max_difference, 6))
