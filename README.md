# Delhivery Delay Intelligence

## Project Overview

Delhivery Delay Intelligence is a machine learning project designed to predict shipment delays and support proactive logistics decision-making.

The project analyses shipment details, delivery distances, hub performance, service types and operational factors to identify shipments that are likely to be delayed.

It also segments delivery hubs based on their operational performance and recommends suitable actions for newly arriving shipments.

## Project Objectives

- Analyse historical shipment data and identify delay patterns.
- Clean and prepare shipment data for machine learning.
- Develop regression models to predict delivery time.
- Build classification models to identify delayed shipments.
- Identify important features influencing shipment delays.
- Segment delivery hubs into operational performance groups.
- Generate predictions and recommended actions for new shipments.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Python IDLE
- GitHub

## Repository Structure

```text
Delhivery-Delay-Intelligence/
│
├── README.md
├── requirements.txt
├── generate_data.py
│
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
│
├── src/
│   └── shipment_prep.py
│
├── models/
│   ├── delay_model.pkl
│   ├── delay_threshold.csv
│   ├── feature_order.csv
│   └── impute_values.csv
│
├── outputs/
│   ├── part_a_checks.csv
│   ├── part_a_delay_by_group.csv
│   ├── part_a_correlations.csv
│   ├── part_b_scaling_checks.csv
│   ├── part_b_baseline.csv
│   ├── part_b_tree_depths.csv
│   ├── part_b_model_comparison.csv
│   ├── part_b_gb_learning_rates.csv
│   ├── part_b_top2_cv.csv
│   ├── part_b_final_test.csv
│   ├── part_c_models.csv
│   ├── part_c_feature_scores.csv
│   ├── part_c_k_comparison.csv
│   ├── part_c_tuning_comparison.csv
│   ├── part_c_thresholds.csv
│   ├── part_c_final_config.csv
│   ├── part_c_final_test.csv
│   ├── part_d_hub_profile.csv
│   ├── part_d_k_separation.csv
│   ├── part_d_hub_segments.csv
│   ├── part_d_segment_profile.csv
│   ├── part_d_tier_mismatch.csv
│   ├── part_d_reload_check.csv
│   └── new_shipment_actions.csv
│
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
└── predict_delay.py
```

## Methodology

### Part A: Data Cleaning and Exploratory Data Analysis

The raw shipment dataset contains 1,224 records. Duplicate records were removed, missing values were handled, inconsistent service type spellings were standardised, and invalid parcel weights were treated as missing values.

After cleaning, 1,200 unique shipment records were retained.

Key findings:

- Express shipments had a higher delay rate than Standard and Economy shipments.
- Delay rates increased across hub tiers.
- Longer delivery distances were associated with higher delay rates.
- Distance, promised delivery hours and hub-related factors showed important relationships with delivery time.
- A log transformation was applied to declared shipment value to reduce skewness.

### Part B: Regression Modelling

Regression models were developed to predict shipment delivery hours.

The dataset was divided into training, validation and testing sets using a 70:15:15 split.

Models evaluated included:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The best decision tree depth was 6.

Linear Regression was selected as the winning regression model based on validation performance.

**Final test R²: 0.9493**

This indicates strong predictive performance on the held-out test data.

### Part C: Classification Modelling

Classification models were developed to predict whether a shipment would be delayed.

The target variable was:

- 1 – Delayed
- 0 – On-time

Models and techniques included:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Feature selection using SelectKBest
- Grid Search and Randomized Search
- Classification threshold evaluation

The final selected method was the default Random Forest configuration with a threshold of 0.5.

**Final test F1-score: 0.7209**

The classification model helps identify shipments requiring operational attention.

### Part D: Hub Segmentation

K-Means clustering was applied to hub-level operational metrics.

The analysis considered:

- Average delivery hours
- Delay rate
- Average delivery distance
- Average number of stops

Three hub segments were identified:

1. **Standard:** Hubs with comparatively lower delay rates.
2. **Watch:** Hubs requiring regular monitoring.
3. **Escalate:** Hubs with comparatively higher delay rates and greater operational risk.

The analysis also identified four hubs where actual performance-based segments differed from the expected tier-based classification. This indicates that hub tier alone is not sufficient to explain operational performance.

## Model Packaging and New Shipment Prediction

The final classification model was saved using Joblib.

The prediction process uses:

- The saved classification model
- The selected probability threshold
- The saved feature order
- The saved imputation values
- Hub segment information

The model was tested by reloading the saved model and comparing its predictions with the original model predictions.

The prediction script processed 48 new shipments.

Prediction summary:

- Total new shipments: 48
- Flagged shipments: 8
- Hub manager escalations: 6
- Proactive customer calls: 2

Recommended actions:

- Flagged shipments in Escalate hubs: Escalate to hub manager.
- Other flagged shipments: Proactive customer call.
- Non-flagged shipments: No action.

The final predictions are saved in `outputs/new_shipment_actions.csv`.

## How to Run the Project

Run all commands from the repository's main directory.

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate the dataset

```bash
python generate_data.py
```

### Step 3: Run data preparation and analysis

```bash
python part_a_clean_eda.py
```

### Step 4: Run regression modelling

```bash
python part_b_regression.py
```

### Step 5: Run classification modelling

```bash
python part_c_classification.py
```

### Step 6: Run hub segmentation

```bash
python part_d_segments.py
```

### Step 7: Package the final model

```bash
python package_model.py
```

### Step 8: Predict delays for new shipments

```bash
python predict_delay.py
```

## Operational Recommendations

- Monitor hubs in the Escalate segment closely.
- Use shipment delay probabilities to prioritise operational attention.
- Contact customers proactively when shipments are flagged.
- Escalate high-risk shipments at high-delay hubs to hub managers.
- Monitor distance, service type, hub performance and weather-related factors.
- Regularly evaluate model performance using new delivery outcomes.

## Production Readiness and Monitoring

The project includes reusable data-cleaning functions, saved imputation values, a fixed feature order, a packaged model and a prediction script.

For future deployment:

- Monitor prediction performance using actual delivery outcomes.
- Track precision, recall and F1-score over time.
- Monitor changes in shipment distances, service types and hub performance.
- Retrain the model periodically using recent shipment data.
- Review operational actions and customer outcomes to improve the process.

## Limitations

- The project uses synthetic shipment data.
- Real-world traffic, route disruptions and other external factors may affect delivery times.
- Model performance on synthetic data may differ from real operational performance.
- Hub segments are based on the selected operational metrics and may change over time.
- Predictions indicate risk and do not guarantee that a shipment will be delayed.

## Conclusion

Delhivery Delay Intelligence demonstrates how data analytics and machine learning can support shipment delay prediction, hub performance monitoring and proactive logistics operations.

By combining regression, classification and clustering techniques, the project provides a data-driven approach to improving shipment visibility and operational decision-making.
