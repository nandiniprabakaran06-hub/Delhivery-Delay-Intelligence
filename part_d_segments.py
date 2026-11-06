
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

os.makedirs("outputs", exist_ok=True)

# Read modelling table
df = pd.read_csv("data/modelling_table.csv")

# --------------------------------
# HUB PERFORMANCE PROFILE
# --------------------------------

hub_profile = df.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).reset_index()

hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)

hub_profile.round(4).to_csv(
    "outputs/part_d_hub_profile.csv",
    index=False
)

print("Number of hubs:", len(hub_profile))

# --------------------------------
# K-MEANS CLUSTERING
# --------------------------------

CLUSTER_FEATURES = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    hub_profile[CLUSTER_FEATURES]
)

# --------------------------------
# TEST DIFFERENT K VALUES
# --------------------------------

separation_results = []

for k in [2, 3, 4, 5]:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    distances = pairwise_distances(X_scaled)

    within_distances = []
    across_distances = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):

            if labels[i] == labels[j]:
                within_distances.append(distances[i, j])
            else:
                across_distances.append(distances[i, j])

    within_mean = np.mean(within_distances)
    across_mean = np.mean(across_distances)

    sizes = sorted(
        np.bincount(labels).tolist(),
        reverse=True
    )

    separation_results.append({
        "k": k,
        "cluster_sizes": "-".join(map(str, sizes)),
        "within_mean_distance": within_mean,
        "across_mean_distance": across_mean,
        "separation_ratio": across_mean / within_mean
    })

pd.DataFrame(separation_results).round(4).to_csv(
    "outputs/part_d_k_separation.csv",
    index=False
)

# --------------------------------
# FINAL THREE-CLUSTER SOLUTION
# --------------------------------

final_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = final_kmeans.fit_predict(X_scaled)

# Calculate cluster delay rates
cluster_order = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
    .index.tolist()
)

segment_names = {
    cluster_order[0]: "Standard",
    cluster_order[1]: "Watch",
    cluster_order[2]: "Escalate"
}

hub_profile["segment"] = (
    hub_profile["cluster"].map(segment_names)
)

# Save hub segments
hub_segments = hub_profile[
    ["hub_id", "cluster", "segment"]
].copy()

hub_segments.to_csv(
    "outputs/part_d_hub_segments.csv",
    index=False
)

# --------------------------------
# SEGMENT PERFORMANCE PROFILE
# --------------------------------

segment_profile = hub_profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()

segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = segment_profile.sort_values(
    "segment"
)

segment_profile["segment"] = (
    segment_profile["segment"].astype(str)
)

segment_profile.round(4).to_csv(
    "outputs/part_d_segment_profile.csv",
    index=False
)

# --------------------------------
# COMPARE WITH HUB TIER
# --------------------------------

tier_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

tier_data = (
    df[["hub_id", "hub_tier"]]
    .drop_duplicates()
)

mismatch = hub_profile[
    ["hub_id", "segment"]
].merge(
    tier_data,
    on="hub_id",
    how="left"
)

mismatch["tier_expected_segment"] = (
    mismatch["hub_tier"].map(tier_segment)
)

mismatch["mismatch"] = (
    mismatch["segment"] != mismatch["tier_expected_segment"]
)

mismatch = mismatch[mismatch["mismatch"]].copy()

mismatch.to_csv(
    "outputs/part_d_tier_mismatch.csv",
    index=False
)

# --------------------------------
# COMPLETION MESSAGE
# --------------------------------

print("--------------------------------")
print("PART D COMPLETED SUCCESSFULLY!")
print("Number of hubs:", len(hub_profile))
print("Number of segments:", hub_profile["segment"].nunique())
print("Mismatched hubs:", len(mismatch))
print("Check the outputs folder for CSV files.")
