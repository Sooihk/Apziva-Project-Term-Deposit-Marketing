from term_deposit_marketing.load_data import load_deposit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def convert_object_features(df):
    df = df.copy()

    # Exclude the target variable
    object_cols = df.select_dtypes(include='object')

    # Define binary and categorical features
    binary_features = []
    categorical_features = []

    for col in object_cols:
        unique_vals = df[col].nunique()
        if unique_vals == 2:
            binary_features.append(col)
        else:
            categorical_features.append(col)

    # Convert binary features to 0/1
    for col in binary_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col]).astype('int64')

    # converting categorical features to dummy variables
    df[categorical_features] = df[categorical_features].astype('category')

    return df

def handle_unknown_values(df, lower_thresh=5.0, upper_thresh=30.0):
    df = df.copy()
    total_rows = df.shape[0]

    for col in df.columns:
        if 'unknown' in df[col].values:
            unknown_count = (df[col] == 'unknown').sum()
            unknown_pct = unknown_count / total_rows * 100

            if unknown_pct < lower_thresh:
                df = df[df[col] != 'unknown']  # Drop rows
            elif unknown_pct <= upper_thresh:
                # Impute with mode
                mode_val = df[col].mode()[0]
                df[col] = df[col].replace('unknown', mode_val)
            else:
                # Optionally drop column if unknowns are too frequent
                df.drop(columns=col, inplace=True)

    return df

def encode_categorical_features(df):
    df = df.copy()

    # Ordinal encoding for 'education'
    education_order = ['primary', 'secondary', 'tertiary']
    df['education'] = df['education'].astype(pd.api.types.CategoricalDtype(categories=education_order, ordered=True))
    df['education'] = df['education'].cat.codes

    # One-hot encode 'job' and 'marital'
    df['job'] = df['job'].cat.remove_unused_categories()
    df = pd.get_dummies(df, columns=['job', 'marital'], drop_first=True)

    # Convert 'month' from categorical to string before mapping
    df['month'] = df['month'].astype(str).str.lower()
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month'] = df['month'].map(month_map).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df.drop(columns='month', inplace=True)

    return df

def heatmap_new(term_deposit):
    correlation_deposit = term_deposit.copy()
    correlation_matrix = correlation_deposit.corr()

    # Move 'y' to the end (both rows and columns)
    cols = [col for col in correlation_matrix.columns if col != 'y'] + ['y']
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

# Call the preprocessing and feature engineering functions
if __name__ == "__main__":
    term_deposit = load_deposit()
    term_deposit_changed = convert_object_features(term_deposit)
    term_deposit_unknown_removed = handle_unknown_values(term_deposit_changed)
    term_deposit_preprocessed = encode_categorical_features(term_deposit_unknown_removed)
    heatmap_new(term_deposit_preprocessed)



