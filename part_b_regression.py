
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

# Read modelling table
df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["delivery_hours"]

# Split: 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

# --------------------------------
# SCALING CHECK
# --------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_scaled_max_abs_mean = float(
    np.abs(X_train_scaled.mean(axis=0)).max()
)

test_scaled_nonzero_mean = float(
    np.abs(X_test_scaled.mean(axis=0)).max()
)

pd.DataFrame([{
    "train_scaled_max_abs_mean": train_scaled_max_abs_mean,
    "test_scaled_max_abs_mean": test_scaled_nonzero_mean
}]).round(4).to_csv(
    "outputs/part_b_scaling_checks.csv",
    index=False
)

# --------------------------------
# BASELINE MODEL
# --------------------------------

baseline_prediction = np.repeat(
    y_train.mean(), len(y_val)
)

baseline_metrics = {
    "model": "Baseline_Mean",
    "MAE": mean_absolute_error(y_val, baseline_prediction),
    "MSE": mean_squared_error(y_val, baseline_prediction),
    "RMSE": np.sqrt(mean_squared_error(y_val, baseline_prediction)),
    "R2": r2_score(y_val, baseline_prediction)
}

pd.DataFrame([baseline_metrics]).round(4).to_csv(
    "outputs/part_b_baseline.csv",
    index=False
)

# --------------------------------
# DECISION TREE DEPTH TEST
# --------------------------------

depths = [1, 2, 3, 4, 6, 8, 12, None]

depth_results = []

for depth in depths:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_prediction)
    )

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_prediction)
    )

    depth_results.append({
        "max_depth": "None" if depth is None else depth,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

depth_table = pd.DataFrame(depth_results)

depth_table = depth_table.sort_values(
    ["val_rmse", "max_depth"],
    ascending=[True, True],
    key=lambda col: (
        col.map(lambda x: 999 if x == "None" else x)
        if col.name == "max_depth" else col
    )
)

depth_table.round(4).to_csv(
    "outputs/part_b_tree_depths.csv",
    index=False
)

best_depth_value = depth_table.iloc[0]["max_depth"]

if best_depth_value == "None":
    best_depth = None
else:
    best_depth = int(best_depth_value)

print("Best decision tree depth:", best_depth)

# --------------------------------
# COMPARE FOUR MODELS
# --------------------------------

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=best_depth,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

model_results = []
fitted_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_val)

    metrics = {
        "model": name,
        "MAE": mean_absolute_error(y_val, prediction),
        "MSE": mean_squared_error(y_val, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_val, prediction)),
        "R2": r2_score(y_val, prediction)
    }

    model_results.append(metrics)
    fitted_models[name] = model

comparison = pd.DataFrame(model_results)

comparison = comparison.sort_values(
    "RMSE"
).reset_index(drop=True)

comparison.round(4).to_csv(
    "outputs/part_b_model_comparison.csv",
    index=False
)

# Select winner using validation RMSE
winner_name = comparison.iloc[0]["model"]
winner_model = fitted_models[winner_name]

print("Best regression model:", winner_name)

# --------------------------------
# GRADIENT BOOSTING LEARNING RATE
# --------------------------------

learning_rate_results = []

for rate in [0.01, 0.1, 1.0]:

    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=rate,
        random_state=42
    )

    gb.fit(X_train, y_train)

    prediction = gb.predict(X_val)

    learning_rate_results.append({
        "learning_rate": rate,
        "RMSE": np.sqrt(
            mean_squared_error(y_val, prediction)
        ),
        "R2": r2_score(y_val, prediction)
    })

pd.DataFrame(learning_rate_results).round(4).to_csv(
    "outputs/part_b_gb_learning_rates.csv",
    index=False
)

# --------------------------------
# CROSS VALIDATION FOR TOP TWO
# --------------------------------

top_two_names = comparison.head(2)["model"].tolist()

X_train_val = pd.concat(
    [X_train, X_val],
    axis=0
)

y_train_val = pd.concat(
    [y_train, y_val],
    axis=0
)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name in top_two_names:

    model = models[name]

    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=cv,
        scoring="r2"
    )

    cv_results.append({
        "model": name,
        "cv_r2_mean": scores.mean(),
        "cv_r2_std": scores.std()
    })

pd.DataFrame(cv_results).round(4).to_csv(
    "outputs/part_b_top2_cv.csv",
    index=False
)

# --------------------------------
# FINAL TEST EVALUATION
# Test set is used only for the winner
# --------------------------------

winner_model.fit(X_train, y_train)

test_prediction = winner_model.predict(X_test)

final_test_result = {
    "model": winner_name,
    "MAE": mean_absolute_error(y_test, test_prediction),
    "MSE": mean_squared_error(y_test, test_prediction),
    "RMSE": np.sqrt(
        mean_squared_error(y_test, test_prediction)
    ),
    "R2": r2_score(y_test, test_prediction)
}

pd.DataFrame([final_test_result]).round(4).to_csv(
    "outputs/part_b_final_test.csv",
    index=False
)

print("--------------------------------")
print("PART B COMPLETED SUCCESSFULLY!")
print("Winning model:", winner_name)
print("Final test R2:", round(final_test_result["R2"], 4))
print("Check the outputs folder for CSV files.")
