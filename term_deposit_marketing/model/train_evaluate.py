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
    """
    # Smote to synthesize new minority class samples


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
