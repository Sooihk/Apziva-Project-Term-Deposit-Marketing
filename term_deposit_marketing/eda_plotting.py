from term_deposit_marketing.load_data import load_deposit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
# heatmap ofr 
def heatmap1(term_deposit):
    correlation_deposit = term_deposit.copy()
    correlation_deposit['y'] = correlation_deposit['y'].map({'yes':1, 'no':0})
    correlation_matrix = correlation_deposit.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10,7))

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
    # Set Title
    ax.set_title("Correlation Heatmap", fontsize=16)
    # Adjust layout to prevent clipping
    fig.tight_layout()
    # Show plot
    plt.show()


# Updated function with bright blue QQ scatterpoints and custom KDE line
def check_skewness_and_qqplots(term_deposit, numerical_cols):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2,5, figsize=(22,8))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        skew_val = term_deposit[col].skew()

        # Histogram with KDE and custom colors
        sns.histplot(term_deposit[col], bins=20, kde=True, ax=axes[i],
                     color='deepskyblue', edgecolor='black', stat='count')

        # Modify KDE line color manually if present
        if axes[i].lines:
            axes[i].lines[0].set_color('orange')  # change KDE line color

        axes[i].set_title(f'{col} | Skew: {skew_val:.2f}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, linestyle='--', alpha=0.6)

        # QQ Plot with bright blue scatterpoints and crimson line
        (osm, osr), (slope, intercept, r) = stats.probplot(term_deposit[col], dist="norm")

        axes[i+5].scatter(osm, osr, color='deepskyblue', alpha=0.6)
        axes[i+5].plot(osm, slope * osm + intercept, color='crimson', linewidth=2)
        axes[i+5].set_title(f'{col} | QQ Plot', fontsize=12, fontweight='bold')
        axes[i+5].set_xlabel("Theoretical Quantiles")
        axes[i+5].set_ylabel("Sample Quantiles")
        axes[i+5].grid(True, linestyle='--', alpha=0.6)

    # Adjust layout
    fig.suptitle("Distribution and Normality Checks for Numerical Features", fontsize=16, fontweight='bold')
    #fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.text(0.5, -0.01, 'Figure: Histograms (top) and Q-Q Plots (bottom) for each feature', 
             ha='center', fontsize=11, style='italic')
    plt.tight_layout()
    plt.show()

# function to plot bar and pie charts for categorical features in the dataset
def plot_categorical_eda(term_deposit ,categorical_cols):
    """
    Plots bar and pie charts for categorical features in a 5x4 grid layout.
    Parameters:
    term_deposit : 
        The dataset containing categorical features.
    categorical_cols : 
        List of categorical column names to plot 
    """
    color_palette = sns.color_palette('pastel')
    # number of rows in dataset used to compute percentages
    total = len(term_deposit)
    num_features = len(categorical_cols)

    # Set up the 5x4 grid (each feature takes 2 plots: bar and pie)
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 22))
    axes = axes.flatten()

    # Main Plotting Loop
    for i, col in enumerate(categorical_cols):
        # assign bar+pie plots next to each other, 
        bar_ax = axes[2*i]
        pie_ax = axes[2*i+1]

        # Count of each category, preprocessing per column
        value_counts = term_deposit[col].value_counts()
        percentages = (value_counts / total * 100).round(1)

        # Bar Plot Construction
        sns.countplot(
            data=term_deposit,
            x=col,
            order=value_counts.index,
            palette=color_palette,
            ax = bar_ax,
            edgecolor='black'
        )
        # Add percentage labesl above bars
        for j, count in enumerate(value_counts):
            bar_ax.text(j, count + total * 0.01, f'{(count / total * 100):.1f}%', 
                        ha='center', fontsize=9)
        bar_ax.set_title(f'{col.capitalize()} – Distribution', fontsize=13, fontweight='bold')
        bar_ax.set_xlabel(col.capitalize(), fontsize=11)
        bar_ax.set_ylabel("Count", fontsize=11)
        bar_ax.tick_params(axis='x', rotation=45)

        # ---- Pie Chart ----
        pie_ax.pie(
            value_counts,
            labels=value_counts.index,
            autopct='%1.1f%%',
            pctdistance=0.8,
            startangle=140,
            colors=color_palette,
            wedgeprops={'edgecolor': 'black'}
        )
        pie_ax.set_title(f'{col.capitalize()} – Proportion', fontsize=13, fontweight='bold')
        pie_ax.axis('equal')

    # Remove unused axes if any
    for idx in range(2 * num_features, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("Categorical Feature Distributions & Proportions", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.show()
        
def boxplot_outliers(term_deposit, numerical_cols):
    sns.set_theme(style='whitegrid')
    
    #plt.figure(figsize=(8,4))
    num_features = len(numerical_cols)
    # create subplots, arrange boxplots horizontally (1 row, n columns)
    # each subplot is width 4, height 6, and sharey=False means each plot has it owns y-axis
    fig, axs = plt.subplots(1,num_features, figsize=(4 * num_features, 6), sharey=False)
    for i, num in enumerate(numerical_cols):
        sns.boxplot(y=term_deposit[num], ax=axs[i], color = 'skyblue', linewidth=1.5)
        axs[i].set_title(f"{num.capitalize()}", fontsize=12, fontweight='bold')
        axs[i].set_ylabel('Value', fontsize=10)
    fig.suptitle('Boxplots for Numerical Features with Outliers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Call the plot
if __name__ == "__main__":
    term_deposit = load_deposit()
    categorical_cols = term_deposit.select_dtypes(include=['object']).columns
    numerical_cols = term_deposit.select_dtypes(include=['int64']).columns
    heatmap1(term_deposit)
    check_skewness_and_qqplots(term_deposit, numerical_cols)
    plot_categorical_eda(term_deposit, categorical_cols)
    boxplot_outliers(term_deposit, numerical_cols)

    