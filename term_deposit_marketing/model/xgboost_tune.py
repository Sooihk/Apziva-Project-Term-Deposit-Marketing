from pathlib import Path
# Imports for modeling, evaluation, and utilities
import os
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split

# Define stratified cross-validation strategy to preserve class balance
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=369)

def resample_training_data(X_train, y_train, random_state=369):
    # Apply SMOTE (oversampling) followed by Tomek Links (undersampling)
    smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=random_state)
    smt = SMOTETomek(smote=smote, tomek=TomekLinks(n_jobs=-1), random_state=random_state)
    return smt.fit_resample(X_train, y_train)

def train_xgboost_with_optuna(X_train, y_train):
    best_y_true, best_y_pred = [], []  # Track best true and predicted labels across trials

    def objective(trial):
        nonlocal best_y_true, best_y_pred  # Store best results across all CV folds

        # Define hyperparameter search space for Optuna
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 400, 800]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1, 5, 10]),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'random_state': 369,
            'n_jobs': -1
        }

        all_y_true, all_y_pred = [], []

        # Perform cross-validation with internal resampling
        for train_idx, val_idx in cv_strategy.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Apply SMOTETomek only on training fold
            X_res, y_res = resample_training_data(X_tr, y_tr, random_state=369)

            model = XGBClassifier(**params)
            model.fit(X_res, y_res)

            # Predict on untouched validation fold
            probas = model.predict_proba(X_val)[:, 1]
            preds = (probas >= 0.5).astype(int)

            all_y_true.extend(y_val)
            all_y_pred.extend(preds)

        # Update best trial predictions
        f1 = f1_score(all_y_true, all_y_pred)
        if f1 > f1_score(best_y_true, best_y_pred):
            best_y_true, best_y_pred = all_y_true, all_y_pred

        return f1  # Maximize F1 score for class 1

    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    # Print best trial results
    print("\nClassification Report for Best CV Model:")
    print(classification_report(best_y_true, best_y_pred))

    # Train final model on fully resampled training set using best hyperparameters
    best_params = {
        **study.best_params,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'random_state': 369,
        'n_jobs': -1
    }
    best_model = XGBClassifier(**best_params)
    X_res_full, y_res_full = resample_training_data(X_train, y_train, random_state=369)
    best_model.fit(X_res_full, y_res_full)

    # Save model to disk
    os.makedirs("../models", exist_ok=True)
    joblib.dump(best_model, "../models/xgboost_model.pkl")

    print("\nBest F1-Score from CV:", study.best_value)
    print("Best Hyperparameters:", study.best_params)
    return best_model


def tune_threshold_and_save(X_train, y_train):
    # Load trained model
    model = joblib.load("../models/xgboost_model.pkl")
    y_proba_cv = np.zeros(len(y_train))

    # Generate predicted probabilities using cross-validation
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        X_res, y_res = resample_training_data(X_tr, y_tr, random_state=369)
        model.fit(X_res, y_res)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_proba_cv[val_idx] = y_proba  # Store fold predictions in correct position

    # Search for the best decision threshold that maximizes F1
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_thresh, best_f1 = 0.5, 0
    best_preds = None

    for thresh in thresholds:
        preds = (y_proba_cv >= thresh).astype(int)
        score = f1_score(y_train, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            best_preds = preds

    print("\nClassification Report at Best Threshold:")
    print(classification_report(y_train, best_preds))

    # Save threshold to disk
    joblib.dump(best_thresh, "../models/xgboost_threshold.pkl")
    print("\nBest Threshold from CV:", best_thresh, "(F1 =", round(best_f1, 4), ")")
    return best_thresh

def evaluate_model_on_test(X_test, y_test, model_path="../models/xgboost_model.pkl", threshold_path="../models/xgboost_threshold.pkl"):
    # Load trained model and best threshold
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)

    # Predict probabilities and classify using tuned threshold
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    # Print classification report and AUC
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("AUC Score:", roc_auc_score(y_test, proba))

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, proba):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent 
    file_name = 'preprocessed_data.csv'
    file_path = base_dir.parent.parent / 'data' / 'interim' /  file_name 
    # obtain transformed dataset csv file
    transformed_df = pd.read_csv(file_path)
    print(transformed_df.info())

    # train/test split transformed dataset
    X = transformed_df.drop(columns='y')
    y = transformed_df['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.20, stratify=y,random_state=369
    )
    best_model = train_xgboost_with_optuna(X_train, y_train)
    best_tuned_model = tune_threshold_and_save(X_train, y_train)
    evaluate_model_on_test(X_test, y_test)