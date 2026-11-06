
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

# Read data
df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]

# Stratified train, validation and test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42,
    stratify=y_temp
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

# Metrics helper
def get_metrics(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        )
    }


# --------------------------------
# BASELINE MODEL
# --------------------------------

baseline_prediction = np.zeros(len(y_val), dtype=int)

baseline_metrics = get_metrics(y_val, baseline_prediction)
baseline_metrics["model"] = "Blind_Baseline"

# --------------------------------
# THREE CLASSIFICATION MODELS
# --------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

model_results = [baseline_metrics]

for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)

    result = get_metrics(y_val, prediction)
    result["model"] = name
    model_results.append(result)

pd.DataFrame(model_results)[
    ["model", "TN", "FP", "FN", "TP",
     "accuracy", "precision", "recall", "f1"]
].round(4).to_csv(
    "outputs/part_c_models.csv", index=False
)

# --------------------------------
# DECISION TREE DEPTH SELECTION
# --------------------------------

tree_results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)
    prediction = tree.predict(X_val)

    tree_results.append({
        "max_depth": depth,
        "val_f1": f1_score(
            y_val, prediction, zero_division=0
        )
    })

tree_table = pd.DataFrame(tree_results).sort_values(
    ["val_f1", "max_depth"],
    ascending=[False, True]
)

tree_table.to_csv(
    "outputs/part_c_tree_depths.csv", index=False
)

best_tree_depth = int(tree_table.iloc[0]["max_depth"])

# --------------------------------
# FEATURE IMPORTANCE AND F SCORES
# --------------------------------

forest = RandomForestClassifier(
    n_estimators=100, random_state=42
)
forest.fit(X_train, y_train)

selector_all = SelectKBest(
    score_func=f_classif, k="all"
)
selector_all.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector_all.scores_,
    "rf_importance": forest.feature_importances_
})

feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(ascending=False, method="min")
    .astype(int)
)

feature_scores["is_noise"] = (
    feature_scores["feature"].isin(NOISE_COLUMNS)
)

feature_scores = feature_scores.sort_values(
    "f_rank"
)

feature_scores.round(4).to_csv(
    "outputs/part_c_feature_scores.csv", index=False
)

# --------------------------------
# SELECT BEST NUMBER OF FEATURES
# --------------------------------

k_results = []

for k in [5, 7, 10, 12, 17]:
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_val)

    k_results.append({
        "k": k,
        "val_f1": f1_score(
            y_val, prediction, zero_division=0
        )
    })

pd.DataFrame(k_results).round(4).to_csv(
    "outputs/part_c_k_selection.csv", index=False
)

# Assignment specifies k=7 for subsequent tuning
selector = SelectKBest(f_classif, k=7)
X_train_7 = selector.fit_transform(X_train, y_train)
X_val_7 = selector.transform(X_val)

# --------------------------------
# HYPERPARAMETER TUNING
# --------------------------------

parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=parameter_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)

grid_search.fit(X_train_7, y_train)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=parameter_grid,
    n_iter=9,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=1

)

random_search.fit(X_train_7, y_train)

# Default forest using seven selected features
default_model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
default_model.fit(X_train_7, y_train)

candidates = [
    ("Default", default_model, None, 1),
    ("GridSearch", grid_search.best_estimator_,
     grid_search.best_params_, 135),
    ("RandomSearch", random_search.best_estimator_,
     random_search.best_params_, 45)
]

tuning_results = []

for name, model, params, fits in candidates:
    prediction = model.predict(X_val_7)

    result = get_metrics(y_val, prediction)

    if params is None:
        n_estimators = 100
        max_depth = "None"
        min_samples_leaf = 1
        cv_accuracy = np.nan
        folds = 0
    else:
        n_estimators = params["n_estimators"]
        max_depth = (
            "None" if params["max_depth"] is None
            else params["max_depth"]
        )
        min_samples_leaf = params["min_samples_leaf"]

        if name == "GridSearch":
            cv_accuracy = grid_search.best_score_
        else:
            cv_accuracy = random_search.best_score_

        folds = 5

    tuning_results.append({
        "method": name,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "cv_folds": folds,
        "total_fits": fits,
        "best_cv_accuracy": cv_accuracy,
        "val_f1": result["f1"]
    })

tuning_table = pd.DataFrame(tuning_results)

tuning_table.round(4).to_csv(
    "outputs/part_c_tuning.csv", index=False
)

# Choose highest validation F1.
# In a tie, the order is Default, GridSearch, RandomSearch.
priority = {
    "Default": 0,
    "GridSearch": 1,
    "RandomSearch": 2
}

tuning_table["priority"] = tuning_table["method"].map(priority)

tuning_table = tuning_table.sort_values(
    ["val_f1", "priority"],
    ascending=[False, True]
)

chosen_method = tuning_table.iloc[0]["method"]

chosen_model = {
    "Default": default_model,
    "GridSearch": grid_search.best_estimator_,
    "RandomSearch": random_search.best_estimator_
}[chosen_method]

chosen_params = {
    "Default": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    },
    "GridSearch": grid_search.best_params_,
    "RandomSearch": random_search.best_params_
}[chosen_method]

# --------------------------------
# THRESHOLD SELECTION
# --------------------------------

validation_probabilities = chosen_model.predict_proba(
    X_val_7
)[:, 1]

threshold_results = []

for threshold in np.arange(0.1, 1.0, 0.1):
    prediction = (
        validation_probabilities >= threshold
    ).astype(int)

    result = get_metrics(y_val, prediction)

    threshold_results.append({
        "threshold": round(float(threshold), 1),
        **result
    })

threshold_table = pd.DataFrame(threshold_results)

threshold_table.round(4).to_csv(
    "outputs/part_c_thresholds.csv", index=False
)

eligible = threshold_table[
    threshold_table["recall"] >= 0.80
]

if len(eligible) > 0:
    chosen_threshold = float(
        eligible["threshold"].max()
    )
else:
    chosen_threshold = 0.1

# --------------------------------
# FINAL TEST EVALUATION
# --------------------------------

# Fit final model on training data only
final_selector = SelectKBest(f_classif, k=7)
X_train_final = final_selector.fit_transform(
    X_train, y_train
)
X_test_final = final_selector.transform(X_test)

final_model = RandomForestClassifier(
    n_estimators=chosen_params["n_estimators"],
    max_depth=chosen_params["max_depth"],
    min_samples_leaf=chosen_params["min_samples_leaf"],
    random_state=42
)

final_model.fit(X_train_final, y_train)

test_probabilities = final_model.predict_proba(
    X_test_final
)[:, 1]

test_prediction = (
    test_probabilities >= chosen_threshold
).astype(int)

final_metrics = get_metrics(y_test, test_prediction)

final_metrics["model"] = "Final_Random_Forest"
final_metrics["threshold"] = chosen_threshold

pd.DataFrame([final_metrics]).round(4).to_csv(
    "outputs/part_c_final_test.csv", index=False
)

# Save final model configuration
config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": (
        "None" if chosen_params["max_depth"] is None
        else chosen_params["max_depth"]
    ),
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_method
}])

config.to_csv(
    "outputs/part_c_final_config.csv", index=False
)

print("--------------------------------")
print("PART C COMPLETED SUCCESSFULLY!")
print("Selected method:", chosen_method)
print("Selected threshold:", chosen_threshold)
print("Final test F1:", round(final_metrics["f1"], 4))
print("Check the outputs folder for CSV files.")
