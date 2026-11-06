
import numpy as np
import pandas as pd

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


def clean_shipments(raw, medians=None):

    df = raw.copy()

    # Remove duplicate rows
    df = df.drop_duplicates().copy()

    # Clean service type
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # Fill missing payment mode
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Clean declared value
    df["declared_value_inr"] = (
        df["declared_value"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value_inr"], errors="coerce"
    )

    df = df.drop(columns=["declared_value"])

    # Convert dispatch time
    df["dispatch_time"] = pd.to_datetime(
        df["dispatch_time"], errors="coerce"
    )

    # Treat unrealistic parcel weights as missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # Calculate or reuse training medians
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median())
        }

    # Fill missing numerical values
    for column, value in medians.items():
        df[column] = df[column].fillna(value)

    return df, medians


def build_features(clean, hubs):

    df = clean.copy()

    # Add hub information
    df = df.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity"
            ]
        ],
        on="hub_id",
        how="left"
    )

    # Promised delivery hours
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)

    # Payment feature
    df["is_cod"] = (
        df["payment_mode"].str.upper() == "COD"
    ).astype(int)

    # Weekend feature
    df["is_weekend"] = (
        df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # Dispatch hour
    df["dispatch_hour"] = df["dispatch_time"].dt.hour

    # Log transformation
    df["log_declared_value"] = np.log1p(
        df["declared_value_inr"]
    )

    return df
    
