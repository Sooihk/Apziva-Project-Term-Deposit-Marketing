from term_deposit_marketing.load_data import load_deposit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pathlib import Path

def convert_object_features(df):
    """ 
    Identify all object type columns and splits them into binary and categorical features.
    Encodes binary features using LabelEncoder to 0 and 1. 
    Casts categorical features to category dtype
    """
    term_df = df.copy()
    # Select all columns in Dataframe that are of Python dtype
    object_features = term_df.select_dtypes(include='object')

    # Define empty binary and and categorical features
    binary_features = []
    categorical_features = []
    # Iterate over all object features and assign features to binary or categorical lists
    for itr in object_features:
        unique_vals = term_df[itr].nunique()
        if unique_vals == 2:
            binary_features.append(itr)
        else:
            categorical_features.append(itr)
    # For each binary feature, convert to 0/1
    for itr in binary_features:
        lEncode = LabelEncoder()
        term_df[itr] = lEncode.fit_transform(term_df[itr]).astype('int64')

    # convert each categorical feature to pandas 'category' type
    term_df[categorical_features] = term_df[categorical_features].astype('category')
    return term_df

def handle_unknown_values(df, lower_thresh=5.0, upper_thresh=30.0, verbose=True):
    """ 
    For each column that contains unknown values:
    If <5% of the values are unknown, drop these rows
    If between 5%-30%, replace these with the mode of that column
    If >30%, drop the entire column as noise
    """
    term_df = df.copy()
    # store number of rows in the dataframe for percentage calculations
    total_rows = term_df.shape[0]
    # counter to keep track of how many rows were dropped
    rows_dropped = 0
    columns_dropped = []

    for itr in term_df.columns:
        # check each column to see if string 'unknown' is present in any row 
        if 'unknown' in term_df[itr].values:
            # Count how many unknowns in this column
            unknown_count = (term_df[itr] == 'unknown').sum()
            unknown_percentage = unknown_count / total_rows * 100

            if unknown_percentage < lower_thresh:
                # Drop rows
                term_df = term_df[term_df[itr] != 'unknown']
                rows_dropped+=unknown_count
                if verbose:
                    print(f"Dropped {unknown_count} rows from column '{itr}' (<{lower_thresh}% unknown)")

            elif unknown_percentage <= upper_thresh:
                # Impute with mode
                mode_val = term_df[itr].mode()[0]
                term_df[itr] = term_df[itr].replace('unknown', mode_val)
                if verbose:
                    print(f"Imputed 'unknown' in column '{itr}' with mode: {mode_val} ({unknown_percentage:.2f}%)")
            # If unknown values are above the threshold
            else:
                term_df.drop(columns=itr, inplace=True)
                columns_dropped.append(itr)
                if verbose:
                    print(f"Dropped column '{itr}' (> {upper_thresh}% unknown)")
    if verbose:
        print(f"\nSummary: Dropped {rows_dropped} rows, Dropped columns: {columns_dropped}")
    return term_df

                
def encode_categorical_features(df):
    """ 
    Ordinal Encoding of 'education': primary=0,secondary=1,tertiary=2
    One hot encoding of 'job' and 'martial'
    Cyclical encoding of 'month' into month_sin and month_cos to preserve cyclical nature
    """
    term_df = df.copy()

    # Ordinal encoding for 'education'
    education_order = ['primary', 'secondary', 'tertiary']
    # Convert education col to pandas Categorical type with specified order
    term_df['education'] = term_df['education'].astype(
        pd.api.types.CategoricalDtype(categories=education_order, ordered=True))
    # Replace education values with integer codes based on defined order
    term_df['education'] = term_df['education'].cat.codes

    # remove any unused categories in job
    term_df['job'] = term_df['job'].cat.remove_unused_categories()
    term_df = pd.get_dummies(term_df, columns=['job', 'marital'], drop_first=True)

    # Convert month to lower case string for uniformity
    term_df['month'] = term_df['month'].astype(str).str.lower()
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    # Replace month strings to numerical equivalents using map dictionary
    term_df['month'] = term_df['month'].map(month_map).astype(int)
    # sin and cos preserve periodicity of time
    term_df['month_sin'] = np.sin(2 * np.pi * term_df['month'] / 12)
    term_df['month_cos'] = np.cos(2 * np.pi * term_df['month'] / 12)
    # Remove original month column
    term_df.drop(columns = 'month', inplace=True)

    return term_df

def heatmap_new(term_deposit):
    correlation_deposit = term_deposit.copy()
    correlation_matrix = correlation_deposit.corr()

    # Reorder columns and rows to move 'y' to the end
    # .tolist() + ['y'] adds 'y' to the end
    cols = correlation_matrix.columns.drop('y').tolist() + ['y']
    # .loc[cols,cols] reorders both rows and columns
    correlation_matrix = correlation_matrix.loc[cols, cols]

    fig, ax = plt.subplots(figsize=(20,14))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidth=0.5,
        square=True,
        cbar_kws={'shrink':0.8},
        ax=ax
    )

    ax.set_title("Correlation Heatmap", fontsize=16)
    fig.tight_layout()
    plt.show()

def drop_low_correlation_features(df, target_column='y', threshold=0.01):
    """
    Drops features from df that have low linear correlation with the target variable.
    """
    # Pearson correlation matrix in dataframe
    corr_matrix = df.corr()
    # Extract correlation between each feature and target variable
    target_corr = corr_matrix[target_column].drop(target_column)

    # Remove features where correlation with target between -threshold and threshold
    low_corr_features = target_corr[(-threshold < target_corr)&(target_corr < threshold)].index.tolist()
    df_filtered = df.drop(columns=low_corr_features)

    print(f"Dropped {len(low_corr_features)} features with |correlation| < {threshold}:")
    print(low_corr_features)

    return df_filtered

def cap_outliers_iqr(series, name=None):
    """ 
    Cap outlier values in a numerical column using the Interquantile Range (IQR) method.
    """
    # Get the 25th and 75th percentile
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    # Compute Interquantile Range
    iqr = q3 - q1
    # Outlier boundaries
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    # Clip values outside the boundaries
    clipped_values = np.clip(series, lower, upper)
    # print how many values were capped in a column
    if name:
        amount_capped = ((series < lower) | (series > upper)).sum()
        print(f'{name} : {clipped_values} values were capped using IQR')
    # return transformed clipped series
    return clipped_values
def transform_skewed_features(df):
    """ 
    Log transform and cap outlers in skewed numeric features
    """
    transformed_df = df.copy()
    # Shift balance to ensure positivity for log transformation
    if (transformed_df['balance'] <= 0).any():
        transformed_df['balance'] = transformed_df['balance'] + abs(transformed_df['balance'].min()) + 1
    # Apply natural log to reduce skew and compress large values
    transformed_df['balance'] = np.log(transformed_df['balance'])

    # Apply log and add 1 to avoid log 0
    transformed_df['duration'] = np.log(transformed_df['duration'] + 1)

    # For campaign feature, cap values greater than 10 to 10 to limit extreme calls
    transformed_df['campaign'] = np.where(transformed_df['campaign'] > 10, 10, transformed_df['campaign'])
    # Log transform
    transformed_df['campaign'] = np.log(transformed_df['campaign'] + 1)

    # Cap outliers after log transformations
    transformed_df['balance'] = cap_outliers_iqr(transformed_df['balance'], name='balance')
    transformed_df['duration'] = cap_outliers_iqr(transformed_df['duration'], name = 'duration')
    transformed_df['campaign'] = cap_outliers_iqr(transformed_df['campaign'], name = 'campaign')

    return transformed_df


# Call the preprocessing and feature engineering functions
if __name__ == "__main__":
    print("\nLoading raw dataset...")
    term_deposit = load_deposit()
    print("Converting object-type features: encoding binary columns to 0/1 and converting others to categorical types...")
    term_deposit_changed = convert_object_features(term_deposit)
    print("Handling 'unknown' values: dropping rows, imputing with mode, or dropping columns based on frequency thresholds...")
    term_deposit_unknown_removed = handle_unknown_values(term_deposit_changed)
    print("Encoding categorical features: ordinal encoding for 'education', one-hot encoding for 'job' and 'marital', and cyclical transformation for 'month'")
    term_deposit_preprocessed = encode_categorical_features(term_deposit_unknown_removed)
    heatmap_new(term_deposit_preprocessed)
    term_deposit_changed_v2 = drop_low_correlation_features(term_deposit_preprocessed)
    term_deposit_changed_v3 = transform_skewed_features(term_deposit_changed_v2)
    # save dataframe to data path
    # Get path relative to current file's directory
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / 'data' / 'interim'
    data_path.mkdir(parents=True, exist_ok=True)

    file_path = data_path / 'preprocessed_data.csv'

    term_deposit_changed_v3.to_csv(file_path, index=False)
    print(f'Transformed data saved to : {file_path}')




