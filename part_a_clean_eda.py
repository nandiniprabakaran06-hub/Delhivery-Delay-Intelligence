
import os
import pandas as pd

from src.shipment_prep import (
    clean_shipments,
    build_features,
    FEATURES
)

# Create output folder if needed
os.makedirs("outputs", exist_ok=True)

# Read the generated data
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Record original data quality details
raw_rows = len(raw)
duplicate_rows = int(raw.duplicated().sum())

missing_distance = int(raw["distance_km"].isna().sum())
missing_weight = int(raw["parcel_weight_kg"].isna().sum())
missing_value = int(raw["declared_value"].isna().sum())
missing_payment = int(raw["payment_mode"].isna().sum())

sentinel_weight = int((raw["parcel_weight_kg"] > 100).sum())

service_spellings_before = int(raw["service_type"].nunique())

# Clean the shipment data
clean, medians = clean_shipments(raw)

service_spellings_after = int(clean["service_type"].nunique())

# Build model features and merge hub details
model_table = build_features(clean, hubs)

# Create delay target
model_table["is_delayed"] = (
    model_table["delivery_hours"] > model_table["promised_hours"]
).astype(int)

# Save modelling table
model_table.to_csv("data/modelling_table.csv", index=False)

# -------------------------------
# PART A: DATA QUALITY CHECKS
# -------------------------------

checks = []

def add_check(name, value):
    checks.append({
        "check": name,
        "value": value
    })

add_check("raw_rows", raw_rows)
add_check("duplicate_rows_removed", duplicate_rows)
add_check("rows_after_dedup", len(clean))
add_check("missing_distance_km_before", missing_distance)
add_check("missing_parcel_weight_before", missing_weight)
add_check("sentinel_weight_rows", sentinel_weight)
add_check("missing_declared_value_before", missing_value)
add_check("missing_payment_mode_before", missing_payment)
add_check("service_type_spellings_before", service_spellings_before)
add_check("service_type_spellings_after", service_spellings_after)

for column, value in medians.items():
    add_check("median_" + column, round(value, 4))

add_check("rows_after_hub_merge", len(model_table))

null_count = int(
    model_table[FEATURES + ["delivery_hours", "is_delayed"]]
    .isna().sum().sum()
)
add_check("nulls_in_features_and_target", null_count)

delayed_count = int(model_table["is_delayed"].sum())
delay_rate = model_table["is_delayed"].mean()

add_check("delayed_count", delayed_count)
add_check("delay_rate", round(delay_rate, 4))

# Outlier calculation using IQR
q1 = model_table["delivery_hours"].quantile(0.25)
q3 = model_table["delivery_hours"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_count = int(
    (
        (model_table["delivery_hours"] < lower_bound) |
        (model_table["delivery_hours"] > upper_bound)
    ).sum()
)

add_check("delivery_hours_q1", round(q1, 4))
add_check("delivery_hours_q3", round(q3, 4))
add_check("outlier_lower_bound", round(lower_bound, 4))
add_check("outlier_upper_bound", round(upper_bound, 4))
add_check("delivery_hours_outlier_count", outlier_count)

add_check(
    "declared_value_mean",
    round(model_table["declared_value_inr"].mean(), 4)
)
add_check(
    "declared_value_median",
    round(model_table["declared_value_inr"].median(), 4)
)
add_check(
    "log_declared_value_mean",
    round(model_table["log_declared_value"].mean(), 4)
)
add_check(
    "log_declared_value_median",
    round(model_table["log_declared_value"].median(), 4)
)

pd.DataFrame(checks).to_csv(
    "outputs/part_a_checks.csv",
    index=False
)

# -------------------------------
# PART A: DELAY BY GROUP
# -------------------------------

group_results = []

def add_group_results(group_type, group_column):
    grouped = model_table.groupby(
        group_column,
        observed=True
    ).agg(
        shipments=("is_delayed", "size"),
        delayed_count=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean")
    ).reset_index()

    for _, row in grouped.iterrows():
        group_results.append({
            "group_type": group_type,
            "group": str(row[group_column]),
            "shipments": int(row["shipments"]),
            "delayed_count": int(row["delayed_count"]),
            "delay_rate": round(row["delay_rate"], 4)
        })

# Service type
add_group_results("service_type", "service_type")

# Hub tier
add_group_results("hub_tier", "hub_tier")

# Distance bands
model_table["distance_band"] = pd.cut(
    model_table["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"]
)

add_group_results("distance_band", "distance_band")

pd.DataFrame(group_results).to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False
)

# -------------------------------
# PART A: CORRELATION ANALYSIS
# -------------------------------

numeric_columns = model_table.select_dtypes(
    include="number"
).columns

correlation_delivery = (
    model_table[numeric_columns]
    .corr()["delivery_hours"]
    .drop("delivery_hours")
)

correlation_delay = (
    model_table[numeric_columns]
    .corr()["is_delayed"]
    .drop("is_delayed")
)

correlation_results = pd.DataFrame({
    "feature": correlation_delivery.index,
    "corr_delivery_hours": correlation_delivery.values
})

correlation_results["corr_is_delayed"] = (
    correlation_results["feature"]
    .map(correlation_delay)
)

correlation_results = correlation_results.round(4)

correlation_results.to_csv(
    "outputs/part_a_correlations.csv",
    index=False
)

print("PART A COMPLETED SUCCESSFULLY!")
print("Rows in modelling table:", len(model_table))
print("Delayed shipments:", delayed_count)
print("Overall delay rate:", round(delay_rate, 4))
print("Medians:", medians)
print("Check the data and outputs folders for CSV files.")
