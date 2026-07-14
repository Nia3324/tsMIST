import pandas as pd
from scipy.stats import wilcoxon


df_linear = pd.read_csv('final/linear/all_linear.csv') 
df_pathi = pd.read_csv('final/pathi/all_dba.csv')

group_cols = ['dataset', 'model', 'class']
metrics = ['mean_of_metric_mean', 'mean_of_metric_std']

# Grouping
df_linear_grouped = df_linear.groupby(group_cols)[metrics].mean().reset_index()
df_pathi_grouped = df_pathi.groupby(group_cols)[metrics].mean().reset_index()

print(f"Linear groups: {len(df_linear_grouped)}, PathI groups: {len(df_pathi_grouped)}")

# Merging (inner join drops any (dataset, model, class) not present in both)
merged = pd.merge(
    df_linear_grouped, 
    df_pathi_grouped, 
    on=group_cols, 
    suffixes=('_linear', '_pathi')
)
print(f"Matched pairs used in test: {len(merged)}")

# Raw values, no transform: mean is not "lower is better", std is.
raw_mean_linear = merged['mean_of_metric_mean_linear']
raw_mean_pathi = merged['mean_of_metric_mean_pathi']

raw_std_linear = merged['mean_of_metric_std_linear']
raw_std_pathi = merged['mean_of_metric_std_pathi']

# --- Statistical Testing ---
print(f"{'Metric':<25} | {'p-value':<10} | {'Significant'}")
print("-" * 50)

tests = [
    ('Mean', raw_mean_linear, raw_mean_pathi),
    ('Std_Deviation', raw_std_linear, raw_std_pathi)
]

for name, col_lin, col_path in tests:
    stat, p_value = wilcoxon(col_lin, col_path, zero_method='zsplit')
    sig = "Yes" if p_value < 0.05 else "No"
    median_diff = (col_path - col_lin).median()

    print(f"{name:<25} | {p_value}     | {sig}")
    if name == 'Mean':
        print(f"    -> Median Difference (PathI - Linear): {median_diff}")
        print(f"       (Not a 'better/worse' comparison: raw tsMIST_Avg is a boundary-position")
        print(f"        estimate, not a lower-is-better score; a difference indicates the two")
        print(f"        operators locate the boundary differently, not that one is superior.)")
    else:
        print(f"    -> Median Difference (PathI - Linear): {median_diff} (Negative = PathI more consistent)")