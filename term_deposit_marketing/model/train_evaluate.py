from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

def feature_selection(X_train, y_train):
    """ 
    Performs feature selectio using a combination of filter, wrapper and embedded methods. 
    Returning a Dataframe with importance scores and ranks from multiple selection strategies for 
    each feature in the training data.
    """
    feature_names = X_train.columns     # extract feature names
    results = pd.DataFrame(index=feature_names)     # initialize empty Dataframe with feature names as index to store scores for each selection method

    # Filter methods, rely on statisitcal characteristics of the data, independent of any model
    # Compute Mutual Information between features and target
    mi_scores = mutual_info_classif(X_train, y_train, random_state=369)
    results['Mutual_Info'] = pd.Series(mi_scores, index=feature_names)

    # Compute ANOVA F-test scores, selects features using univariate linear regression F-stats
    #  Evaluate if group means for the feature differ significiantly between target classes
    f_test = SelectKBest(score_func=f_classif, k='all')
    f_test.fit(X_train, y_train)
    results['ANOVA_F_test'] = pd.Series(f_test.scores_, index = feature_names)
    #---------------------------------------------------------------------------------------------------------------------

    # Wrapper Method, evaluate feature subset based on model performance
    # Recursive Feature Elimination 
    recursive_feature_elimination = RFE(estimator=LogisticRegression(solver='liblinear', random_state=369), n_features_to_select=20)
    recursive_feature_elimination.fit(X_train, y_train)
    results['RFE_selected'] = recursive_feature_elimination.support_.astype(int)
    #---------------------------------------------------------------------------------------------------------------------

    # Embedded Method
    # L1 Regularization (Lasso), train data on Logistic Regression with L1 regularization
    # storing absolute value of each coef as the importance score
    lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=369, max_iter=1000)
    lasso.fit(X_train, y_train)
    results['Lasso_coef'] = pd.Series(np.abs(lasso.coef_).flatten(), index=feature_names)

    # Random Forest Feature Importance
    random_forest = RandomForestClassifier(n_estimators=100,random_state=369)
    # Train random forest and store Gini-based importance scores, reflecting how useful each feature was across trees
    random_forest.fit(X_train, y_train)
    results['RF_importance'] = pd.Series(random_forest.feature_importances_, index=feature_names)

    # XGBoost Feature Importance
    # Initialize XGBoost classifier, train on dataset and extract feature importance scores
    xg_Boost = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=369)
    xg_Boost.fit(X_train, y_train)
    results['XGB_importance'] = pd.Series(xg_Boost.feature_importances_, index=feature_names)

    # Rank each column (higher = more important)
    for col in results.columns:
        if col != 'RFE_selected':
            results[f'{col}_rank'] = results[col].rank(ascending=False)

    return results.sort_values('Mutual_Info', ascending=False)

def plot_feature_rank_heatmap(results_df, top_n=17):
    # Identify rank columns
    rank_cols = [col for col in results_df.columns if col.endswith("_rank")]

    # Sort by Mutual_Info_rank and take top_n
    ranked = results_df[rank_cols].sort_values("Mutual_Info_rank").head(top_n)

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(ranked, annot=True, fmt=".0f", cmap="Blues_r", cbar=True)
    plt.title(f"Feature Ranking Heatmap (Top {top_n})")
    plt.tight_layout()
    plt.show()

def plot_top_features(results, top_k=17):
    """Plot top_k features from each selection method."""
    importance_cols = ['Mutual_Info', 'ANOVA_F_test', 'Lasso_coef', 'RF_importance', 'XGB_importance', 'RFE_selected']
    melted = results[importance_cols].reset_index().melt(id_vars='index', var_name='Method', value_name='Score')
    
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
    axs = axs.flatten()

    for i, method in enumerate(importance_cols):
        subset = melted[melted['Method'] == method].nlargest(top_k, 'Score')
        sns.barplot(data=subset, y='index', x='Score', ax=axs[i], hue='index', palette='Set2', legend=False)
        axs[i].set_title(f"Top {top_k} Features - {method}", fontsize=12, fontweight='bold')
        axs[i].set_xlabel("Score")
        axs[i].set_ylabel("Feature")
        axs[i].grid(axis='x', linestyle='--', alpha=0.5)

    # Hide unused subplot if any
    if len(importance_cols) < len(axs):
        for j in range(len(importance_cols), len(axs)):
            fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def resample_training_data(X_train, y_train, random_state=369):
    """ 
    Apply SMOTETomek resampling on the training data
    Applies SMOTE followed by TomekLinks undersampling.
    Original class distribution:
    - Class 0: 29683 (92.8%)
    - Class 1: 2317 (7.2%)
    """
    # Smote to synthesize new minority class samples
    smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=random_state)
    smote_tomek = SMOTETomek(smote=smote, tomek=TomekLinks(n_jobs=-1), random_state=random_state)
    # Apply resampling only on the training data
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def cross_validated_model_scores(X_train, y_train):
    random_seed=369
    # 5 fold cross validation to perserve class distribution
    cross_validate = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    cross_validate_results = []

    # Dictionary of models to be used, and boolean value if they require scaling
    models = {
        'Logistic Regression': (LogisticRegression(random_state=random_seed, max_iter=1000), True),
        'Random Forest': (RandomForestClassifier(random_state=random_seed), False),
        'XGBoost': (XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0, random_state=random_seed), False),
        'LightGBM': (LGBMClassifier(verbosity=-1, random_state=random_seed), False),
        'Naive Bayes': (GaussianNB(), True),
    }

    # Iterate over each model
    for name, (model, need_scaling) in models.items():
        # Build Pipeline that includesd scaler if needed
        steps = [('scaler', StandardScaler()), ('classifier', model)] if need_scaling else [('classifier', model)]
        pipeline = Pipeline(steps)
        # accumlate validation predictions
        y_true_all = []
        y_pred_all = []
        y_proba_all = []
        # Loop over CV folds, spliting the data into training and validation sets per fold
        for train_index, val_index in cross_validate.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

            # Apply SMOTETomek only to the training fold
            X_resampled, y_resampled = resample_training_data(X_tr, y_tr, random_state=random_seed)
            # fit the model and predict 
            pipeline.fit(X_resampled, y_resampled)
            y_pred = pipeline.predict(X_val) # class labels
            y_proba = pipeline.predict_proba(X_val)[:, 1]  # probability for positive class

            y_true_all.extend(y_val)
            y_pred_all.extend(y_pred)
            y_proba_all.extend(y_proba)
        # Compute metrics (F1, precision, recall) and overall AUC
        report = classification_report(y_true_all, y_pred_all, output_dict=True)
        auc_score = roc_auc_score(y_true_all, y_proba_all)

        cross_validate_results.append({
            'Model': name,
            'CV F1 (1)': report['1']['f1-score'],
            'CV Precision (1)': report['1']['precision'],
            'CV Recall (1)': report['1']['recall'],
            'CV AUC': auc_score
        })
    results_df = pd.DataFrame(cross_validate_results).sort_values(by='CV F1 (1)', ascending=False)
    print("Cross-Validated Model Performance (Training Set Only)")
    print(results_df)
    return results_df

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
    # save the split train/test data into csv files in processed data module
    xtrain_path = base_dir.parent.parent / 'data' / 'processed'
    xtest_path = base_dir.parent.parent / 'data' / 'processed'
    ytrain_path = base_dir.parent.parent / 'data' / 'processed'
    ytest_path = base_dir.parent.parent / 'data' / 'processed'
    xtrain_file_path = xtrain_path / 'xtrain_data.csv'
    xtest_file_path = xtest_path / 'xtest_data.csv'
    ytrain_file_path = ytrain_path / 'ytrain_data.csv'
    ytest_file_path = ytest_path / 'ytest_data.csv'

    X_train.to_csv(xtrain_file_path, index=False)
    X_test.to_csv(xtest_file_path, index=False)
    y_train.to_csv(ytrain_file_path, index=False)
    y_test.to_csv(ytest_file_path, index=False)
    
    print("Running feature selection using filter, wrapper, and embedded methods...")
    feature_selection_results = feature_selection(X_train, y_train)
    plot_feature_rank_heatmap(feature_selection_results, top_n=17)
    plot_top_features(feature_selection_results, top_k=17)
    print("Running cross validation evaluation of multiple models methods...")
    cv_results = cross_validated_model_scores(X_train, y_train)
    best_model = train_xgboost_with_optuna(X_train, y_train)
