import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
from pathlib import Path  


# 1. load data set
base_dir = Path(__file__).resolve().parent 
data_path = base_dir / "accident_prediction_india.csv" # retrive path
output_path = base_dir / "cleaned_data.csv" # upload path

df = pd.read_csv(data_path)  # load data set into pandas





print("First five rows")  # show first five rows
print(df.head(), "\n")

print("Each column")  # show each column
print(df.dtypes, "\n")

print("shape of data set")  
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")


# 2.  analysis Null value
null_counts = df.isnull().sum()  # count missing values
null_percentages = (null_counts / df.shape[0]) * 100  # in percentages

null_info = pd.DataFrame(
    {"Null Count": null_counts, "Null Percentage (%)": null_percentages}
)  
print("null values")
print(null_info, "\n")

# exceeding 20% null rate
high_null_cols = null_percentages[null_percentages > 20].index.tolist()  # find  missing values
print(f"exceed coloumn: {high_null_cols}\n")

# numeric columns below 20% nulls 
low_null_cols = null_percentages[null_percentages <= 20].index  #  columns  acceptable missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns  # select  numeric columns

cols_to_impute = [col for col in numeric_cols if col in low_null_cols]  # select numeric columns 
for col in cols_to_impute:
    median_val = df[col].median() 
    df[col] = df[col].fillna(median_val)  # Replace missing values with the median


# 3. Duplicate Detection and Removal
initial_rows = df.shape[0]  # store number of rows before removing duplicates
duplicate_count = df.duplicated().sum()  # count duplicated rows
print(f"duplicate detection")
print(f"total duplicate rows: {duplicate_count}")

# Capture null percentages before drop to compare impacts
null_perc_before_drop = (df.isnull().sum() / df.shape[0]) * 100  # Save missing-data percentage before cleanup

# Remove duplicates
df = df.drop_duplicates()  # Drop repeated rows from the dataset
rows_removed = initial_rows - df.shape[0]  # Calculate how many rows were removed
print(f"total rows removed: {rows_removed}")

# Check if null percentages changed
null_perc_after_drop = (df.isnull().sum() / df.shape[0]) * 100  # Recalculate missing percentages after dropping rows
null_diff = null_perc_before_drop - null_perc_after_drop  # See how the percentages changed
print("Null percentage change per column after duplicate removal:")
print(null_diff, "\n")


# 4. Data Type Correction & Memory Optimization
mem_before = df.memory_usage(deep=True).sum()  #  memory usage 

for col in df.columns:
    if df[col].dtype == "object":  
        if df[col].nunique(dropna=False) <= 20:  #  unique values
            df[col] = df[col].astype("category")  # turning them into categorical data

mem_after = df.memory_usage(deep=True).sum()  # memory usage after optimization
print("memory optimization")
print(f"memory Before optimization: {mem_before / 1024**2:.2f} MB")
print(f"memory After optimization: {mem_after / 1024**2:.2f} MB")
print(f"saveed: {((mem_before - mem_after) / mem_before) * 100:.2f}%\n")


# 5. Descriptive Statistics & Skewness
print("Descriptive Statistic")  # Show summary statistics for the dataset
print(df.describe(include="all"), "\n")

print("skewed")  # skewed numeric columns 
skew_dict = {}  
numeric_cols_updated = df.select_dtypes(include=[np.number]).columns  # reselect numeric columns after cleaning
for col in numeric_cols_updated:
    skew_val = df[col].skew()  # skewness for each numeric column
    skew_dict[col] = skew_val
    print(f"{col}: {skew_val:.4f}")

if len(skew_dict) > 0:
    highest_skew_col = max(skew_dict, key=lambda k: abs(skew_dict[k]))  # find skewed numeric column
    print(
        f"\ntotal highest skewness: {highest_skew_col} (Skew: {skew_dict[highest_skew_col]:.4f})\n"
    )
else:
    highest_skew_col = None
    print("No numeric columns available for skewness.")


# 6. Outlier Detection with IQR
print("Outlier Detection")  # find unusual values interquartile range
if len(numeric_cols_updated) > 0:
    outlier_target_cols = list(numeric_cols_updated[:2])  # first two numeric columns
    for col in outlier_target_cols:
        Q1 = df[col].quantile(0.25)  # first quartile
        Q3 = df[col].quantile(0.75)  # third quartile
        IQR = Q3 - Q1  # interquartile range
        lower_bound = Q1 - 1.5 * IQR  
        upper_bound = Q3 + 1.5 * IQR 

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]  
        print(f"{col} -> Outliers detected: {len(outliers)} rows")
else:
    print("No numeric columns available for IQR outlier")


# 7. Visualizations
plot_dir = base_dir / "plots"  #generated plots will be saved
plot_dir.mkdir(exist_ok=True)  # if not already exist

plt.figure(figsize=(15, 18))  # multiple charts
available_cat_cols = [
    col
    for col in ["Month", "Day of Week", "Weather Conditions", "City Name", "State Name", "Accident Location Details"]
    if col in df.columns
] 

if available_cat_cols:
    plt.subplot(2, 2, 1)  
    df[available_cat_cols[0]].value_counts().head(10).plot(kind="bar", color="tab:blue")  # Plot first categorical column
    plt.title(f"Count by {available_cat_cols[0]}")
    plt.xlabel(available_cat_cols[0])
    plt.ylabel("Count")

    if len(available_cat_cols) > 1:
        plt.subplot(2, 2, 2)  # Create the second chart area
        df[available_cat_cols[1]].value_counts().head(10).plot(kind="bar", color="tab:orange")  # Plot  second categorical column
        plt.title(f"Count by {available_cat_cols[1]}")
        plt.xlabel(available_cat_cols[1])
        plt.ylabel("Count")

    if len(available_cat_cols) > 2:
        plt.subplot(2, 2, 3)  # Create the third chart area
        df[available_cat_cols[2]].value_counts().head(10).plot(kind="bar", color="tab:green")  # Plot third categorical column
        plt.title(f"Count by {available_cat_cols[2]}")
        plt.xlabel(available_cat_cols[2])
        plt.ylabel("Count")

    if len(available_cat_cols) > 3:
        plt.subplot(2, 2, 4)  # Create the fourth chart area
        df[available_cat_cols[3]].value_counts().head(10).plot(kind="bar", color="tab:red")  # Plot  fourth categorical column
        plt.title(f"Count by {available_cat_cols[3]}")
        plt.xlabel(available_cat_cols[3])
        plt.ylabel("Count")
else:
    plt.text(0.5, 0.5, "No categorical columns available for plotting", ha="center", va="center")  # if there are no suitable columns

plt.tight_layout()  # Arrange subplots neatly
plt.savefig(plot_dir / "categorical_distribution.png", dpi=300, bbox_inches="tight")  #  categorical distribution plot
# plt.show() # to display plot on screen

# 8. Correlation Heat Map
encoded_df = df.copy()  # original data unchanged
for col in encoded_df.select_dtypes(include=['category', 'object', 'string', 'str']).columns:
    encoded_df[col] = encoded_df[col].astype('category').cat.codes  # categories into numbers

if encoded_df.select_dtypes(include=[np.number]).shape[1] >= 2:
    numeric_df = encoded_df.select_dtypes(include=[np.number])  # columns for correlation
    pearson_matrix = numeric_df.corr(method="pearson")  # correlation coefficients

    plt.figure(figsize=(10, 8))  # Create a figure for the heatmap
    sns.heatmap(pearson_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)  # correlation heatmap
    plt.title("Pearson Correlation Coefficient Heatmap")
    plt.tight_layout()
    plt.savefig(plot_dir / "pearson_correlation_heatmap.png", dpi=300, bbox_inches="tight")  # Save heatmap image
    # plt.show() # to display plot on screen

    abs_pearson = pearson_matrix.abs().copy()  # Work with absolute values to find the strongest relationship
    for i in range(len(abs_pearson.columns)):
        abs_pearson.iloc[i, i] = 0  # Remove self-correlation so only pairwise relationships remain
    highest_p_corr = abs_pearson.unstack().idxmax()  # Find the  highest correlation
    print(f"Highest Pearson Correlation Pair: {highest_p_corr}")
    print(f"Value: {pearson_matrix.loc[highest_p_corr[0], highest_p_corr[1]]:.4f}\n")
else:
    print("no numeric columns available for initial column drop.\n")


# Task a: Imputation Strategy Comparison
if len(numeric_cols_updated) > 0:
    top_2_skewed = [
        col for col, _ in sorted(skew_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:2]
    ]  

    print("Imputation Strategy Comparison")  # mean vs median imputation
    for col in top_2_skewed:
        c_mean = df[col].mean()  #  mean for comparison
        c_median = df[col].median()  #  median for comparison
        print(f"Column: {col} | Mean: {c_mean:.4f} | Median: {c_median:.4f}")

        df[col] = df[col].fillna(c_median)  #missing values using the median

    print(f"\nRemaining missing entries in {top_2_skewed}:")
    print(df[top_2_skewed].isnull().sum(), "\n")
else:
    print("No numeric columns available for imputation strategy comparison.\n")


#  Task b: Spearman Rank Correlation
if encoded_df.select_dtypes(include=[np.number]).shape[1] >= 2:
    numeric_df = encoded_df.select_dtypes(include=[np.number])  #  only numeric columns 
    spearman_matrix = numeric_df.corr(method="spearman")  #  Spearman rank correlation
    diff_matrix = (spearman_matrix - pearson_matrix).abs()  #  it against Pearson results

    print("Pearson matrix")  #  the Pearson matrix
    print(pearson_matrix, "\n")
    print("Spearman matrix")  #  the Spearman matrix
    print(spearman_matrix, "\n")

    diff_pairs = diff_matrix.where(
        np.triu(np.ones(diff_matrix.shape), k=1).astype(bool)
    ).unstack()  # Flatten the upper triangle of the difference matrix
    top_3_diffs = diff_pairs.dropna().sort_values(ascending=False).head(3)  # Select the three largest differences

    print("differences between Spearman matrix Pearson")  # Show the biggest differences between methods
    for pair, diff_val in top_3_diffs.items():
        p_val = pearson_matrix.loc[pair[0], pair[1]]  # Pearson value for the pair
        s_val = spearman_matrix.loc[pair[0], pair[1]]  # Spearman value for the pair
        print(
            f"Pair {pair}: |{s_val:.4f} - {p_val:.4f}| = Difference of {diff_val:.4f}"
        )
    print("\n")
else:
    print("No numeric columns available for Spearman correlation analysis.\n")


# Advanced Task c: Grouped Aggregation
if len(numeric_cols_updated) > 0 and available_cat_cols:
    agg_categorical = available_cat_cols[0]  # Choose the first available categorical column for grouping
    agg_numeric = numeric_cols_updated[0]  # Choose the first numeric column for aggregation

    grouped_stats = (
        df.groupby(agg_categorical)[agg_numeric].agg(["mean", "std", "count"]).dropna()
    )  # Group rows by category and calculate summary statistics
    print("--- GROUPED AGGREGATION STATS ---")
    print(grouped_stats, "\n")

    highest_mean = grouped_stats["mean"].max()  # highest group average
    lowest_mean = grouped_stats["mean"].min()  #lowest group average
    mean_ratio = highest_mean / lowest_mean if lowest_mean != 0 else np.nan  # compare the two values
    print(f"Ratio of Highest Group Mean to Lowest Group Mean: {mean_ratio:.2f}\n")
else:
    if available_cat_cols:
        agg_categorical = available_cat_cols[0]
        grouped_stats = df[agg_categorical].value_counts().to_frame("count")
        print("--- GROUPED COUNTS ---")
        print(grouped_stats, "\n")
    else:
        print("No suitable categorical column available for grouped aggregation.\n")





# unwanted columns are removed 
df = df.drop(['Year',
       'Time of Day', 'Accident Severity', 'Number of Vehicles Involved',
       'Vehicle Type Involved', 'Number of Casualties', 'Number of Fatalities',
       'Road Type', 'Road Condition',
       'Lighting Conditions', 'Traffic Control Presence', 'Speed Limit (km/h)',
       'Driver Age', 'Driver Gender', 'Driver License Status',
       'Alcohol Involvement'], axis=1)

# we predicting only Curve and Intersection so Bridge and Stright Road removed 
df= df[df['Accident Location Details'].isin(['Curve','Intersection'])]

# if we remove Unknown then our dataset will be lesser might be face 'underfit' issue so convert 'Unknown' value to 'Some City'
df['City Name']= df['City Name'].replace('Unknown', 'Some City')



# ==========================================
# Save Cleaned Dataset
# ==========================================
df.to_csv(output_path, index=False)
print(f"Successfully saved '{output_path.name}' for Parts 2 and 3.")
