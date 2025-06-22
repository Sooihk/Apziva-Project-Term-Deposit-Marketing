from sklearn.model_selection import train_test_split
from pathlib import Path
# Imports for modeling, evaluation, and utilities
import os
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def prioritize_customer_segments(X, y, model_path="../../models/xgboost_model.pkl", threshold_path="../models/xgboost_threshold.pkl"):
    """
    This function identifies and ranks customer segments that are most likely to purchase a term deposit investment using a trained XGBoost model. 
    It applies probability thresholds, reconstructs categorical labels, and groups buyers by demographic traits.
    """
    # Load the saved XGBoost model and decision threshold to filter likely buyers
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)

    # Copy the input dataset to avoid modifying the original
    demographic_X = X.copy()

    # Predict the probability of subscription for each row in order
    demographic_X['subscription_probability'] = model.predict_proba(demographic_X)[:, 1]

    # Visualize distribution of predicted probabilities and highlight the threshold in order to show how many customers fall above threshold
    plt.figure(figsize=(8, 5))
    plt.hist(demographic_X['subscription_probability'], bins=30, color='skyblue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.title("Distribution of Subscription Probabilities")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Number of Customers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Select only the customers whose predicted probability exceeds the threshold
    likely_buyers = demographic_X[demographic_X['subscription_probability'] >= threshold].copy()

    # Map job and marital group labels directly from one-hot encodings
    job_mapping = {
        'job_blue-collar': 'blue-collar',
        'job_housemaid': 'housemaid',
        'job_management': 'management',
        'job_retired': 'retired',
        'job_services': 'services',
        'job_student': 'student'
    }

    marital_mapping = {
        'marital_married': 'married',
        'marital_single': 'single'
    }

    likely_buyers['job_group'] = None
    for col, label in job_mapping.items():
        if col in likely_buyers.columns:
            likely_buyers.loc[likely_buyers[col] == 1, 'job_group'] = label

    likely_buyers['marital_group'] = None
    for col, label in marital_mapping.items():
        if col in likely_buyers.columns:
            likely_buyers.loc[likely_buyers[col] == 1, 'marital_group'] = label

    # Drop rows where mapping failed (i.e., where no one-hot column was active)
    likely_buyers = likely_buyers.dropna(subset=['job_group', 'marital_group'])

    # Bucketize numeric fields for more actionable segmentation
    likely_buyers['age_group'] = pd.cut(
        likely_buyers['age'],
        bins=[18, 30, 45, 60, 100],
        labels=['18–30', '31–45', '46–60', '60+']
    )
    likely_buyers['balance_group'] = pd.qcut(
        likely_buyers['balance'],
        q=4,
        labels=['low', 'mid-low', 'mid-high', 'high']
    )

    # Group by demographic segments and count how many likely buyers fall into each
    segment_summary = (
        likely_buyers
        .groupby(['job_group', 'marital_group', 'education', 'age_group', 'balance_group'], observed=True)
        .size()
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
    )

    # Output the top segments to prioritize
    print("\nTop Customer Segments Likely to Buy:")
    print(segment_summary.head(10))

    # Plot bar chart of top 10 segments
    top10 = segment_summary.head(10).copy()
    top10['segment'] = top10[['job_group', 'marital_group', 'education', 'age_group', 'balance_group']].astype(str).agg(' | '.join, axis=1)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top10, x='count', y='segment', hue='segment', dodge=False, palette='viridis', legend=False)
    plt.title("Top 10 Customer Segments Likely to Subscribe")
    plt.xlabel("Number of Likely Buyers")
    plt.ylabel("Segment Description")
    plt.tight_layout()
    plt.show()

    return segment_summary

def explain_model_features(model_path="../../models/xgboost_model.pkl", feature_names=None, X_sample=None, top_n=15):

    # Load model
    model = joblib.load(model_path)

    # Get feature importances
    importances = model.feature_importances_
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    # Plot XGBoost feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, y='Feature', x='Importance', palette='viridis')
    plt.title(f"Top {top_n} Most Important Features (XGBoost)", fontsize=14)
    plt.xlabel("Feature Importance (Gain)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Optional: SHAP global summary plots
    if X_sample is not None:
        import shap
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)
        # Bar plot shows the mean absolute SHAP value of each feature
        # Across all customers, how much did this feature change the prediction
        print("\nSHAP Summary Plot (Bar)")
        shap.summary_plot(shap_values, X_sample, show=True, plot_type="bar")
        # color low is blue and red is high, how each feature pushes the prediction up/downm
        print("\nSHAP Summary Plot (Dot)")
        shap.summary_plot(shap_values, X_sample, show=True)

    return importance_df


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent 
    file_name = 'preprocessed_data.csv'
    file_path = base_dir.parent.parent / 'data' / 'interim' /  file_name 
    # obtain transformed dataset csv file
    transformed_df = pd.read_csv(file_path)
    # train/test split transformed dataset
    X = transformed_df.drop(columns='y')
    y = transformed_df['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.20, stratify=y,random_state=369
    )
    model_path = base_dir.parent.parent / 'models' / 'xgboost_model.pkl'
    threshold_path = base_dir.parent.parent / 'models' / 'xgboost_threshold.pkl'
    print(model_path)
    demographic_X = prioritize_customer_segments(X, y, 
                                                 model_path=model_path, threshold_path=threshold_path)
    print(demographic_X.head(20))
    explain_model_features(
    model_path=model_path,
    feature_names=X_test.columns.tolist(),
    X_sample=X_test
    )