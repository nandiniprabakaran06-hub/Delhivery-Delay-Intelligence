
import os
import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# 1. Load the new shipment data and hub details
new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")


# 2. Load the saved model and supporting files
model = joblib.load("models/delay_model.pkl")

threshold = pd.read_csv(
    "models/delay_threshold.csv"
)["threshold"].iloc[0]

feature_order = pd.read_csv(
    "models/feature_order.csv"
)["feature"].tolist()

impute_values_df = pd.read_csv("models/impute_values.csv")

medians = dict(
    zip(impute_values_df["column"], impute_values_df["median"])
)


# 3. Clean the new shipment data using saved medians
cleaned_data, _ = clean_shipments(
    new_shipments,
    medians=medians
)


# 4. Build prediction features
features = build_features(cleaned_data, hubs)

X_new = features[feature_order]


# 5. Predict delay probabilities
probabilities = model.predict_proba(X_new)[:, 1]

results = pd.DataFrame({
    "shipment_id": cleaned_data["shipment_id"].values,
    "hub_id": cleaned_data["hub_id"].values,
    "delay_probability": probabilities.round(4)
})


# 6. Apply the saved threshold
results["flagged"] = (
    results["delay_probability"] >= threshold
).astype(int)


# 7. Add hub segments
hub_segments = pd.read_csv("outputs/part_d_hub_segments.csv")

results = results.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left"
)


# 8. Recommend operational actions
def recommend_action(row):

    if row["flagged"] == 1 and row["segment"] == "Escalate":
        return "Escalate to hub manager"

    elif row["flagged"] == 1:
        return "Proactive customer call"

    else:
        return "No action"


results["action"] = results.apply(
    recommend_action,
    axis=1
)


# 9. Arrange the final output columns
results = results[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
]


# 10. Save prediction results
os.makedirs("outputs", exist_ok=True)

results.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False
)


# 11. Display completion details
print("NEW SHIPMENT PREDICTION COMPLETED SUCCESSFULLY!")
print("Total new shipments:", len(results))
print("Flagged shipments:", int(results["flagged"].sum()))
print(
    "Hub manager escalations:",
    int((results["action"] == "Escalate to hub manager").sum())
)
print("Output saved: outputs/new_shipment_actions.csv")

print("\nFirst 10 shipment predictions:")
print(results.head(10).to_string(index=False))
