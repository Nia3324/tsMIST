import pandas as pd

# 1. Load the data
df = pd.read_csv('final/pathi/all_dba.csv')

# 2. Define mappings
dataset_mapping = {
    "ECG200": "ECG", "EOGHorizontalSignal": "EOGH", "EOGVerticalSignal": "EOGV",
    "Epilepsy": "EPI", "CinCECGTorso": "CET", "ECG5000": "ECG5",
    "ECGFiveDays": "ECGFD", "StandWalkJump": "SWJ", "TwoLeadECG": "TECG",
    "NerveDamage": "ND", "MedicalImages": "MI", "Colposcopy": "COL",
    "EyesOpenShut": "EOS", "ToeSegmentation1": "TOE", "Heartbeat": "HRT",
    "EMOPain": "EMO", "HandMovementDirection": "HMD"
}

# 3. Define columns to aggregate
cols_to_agg = [
    'mean_of_metric_mean', 'std_of_metric_mean',
    'mean_of_metric_std', 'std_of_metric_std',
    'mean_model_acc', 'std_model_acc'
]

# 4. Group and aggregate
grouped = df.groupby(['dataset', 'model'])[cols_to_agg].mean()

# 5. Save the aggregated data to CSV
# 'reset_index()' makes dataset and model regular columns instead of indices
grouped.reset_index().to_csv('aggregated_results.csv', index=False)
print("Aggregated data saved to 'aggregated_results.csv'")

# 6. Pivot for LaTeX table
pivoted = grouped.unstack(level='model')

model_order = ['lstm', 'catch22', 'rocket', 'inception']
model_map = {'lstm': 'LSTM', 'catch22': 'Catch22', 'rocket': 'Rocket', 'inception': 'Incep'}

def format_cell(mean_val, std_val, precision=3):
    return f"{mean_val:.{precision}f}$_{{\\pm {std_val:.2f}}}$"

# 7. Generate and print LaTeX table
print(r"\begin{tabular}{l *{3}{c} | *{3}{c} | *{3}{c} | *{3}{c}}")
print(r"    \toprule")
print(r"    \multirow{2}{*}{\textbf{DS}}")
header_str = "    "
for m in model_order:
    header_str += f"& \multicolumn{{3}}{{c}}{{\textbf{{{model_map[m]}}}}} "
print(header_str + r"\\")
print(r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}")
print(r"    & $M_{Avg}$ & $M_{Std}$ & $Acc$ & $M_{Avg}$ & $M_{Std}$ & $Acc$ & $M_{Avg}$ & $M_{Std}$ & $Acc$ & $M_{Avg}$ & $M_{Std}$ & $Acc$ \\")
print(r"    \midrule")

for dataset, row in pivoted.iterrows():
    label = dataset_mapping.get(dataset, dataset)
    row_str = f"    \\textbf{{{label}}} "
    
    for m in model_order:
        val1 = format_cell(row[('mean_of_metric_mean', m)], row[('std_of_metric_mean', m)], precision=3)
        val2 = format_cell(row[('mean_of_metric_std', m)], row[('std_of_metric_std', m)], precision=3)
        val3 = format_cell(row[('mean_model_acc', m)], row[('std_model_acc', m)], precision=2) 
        
        row_str += f"& {val1} & {val2} & {val3} "
    print(row_str + r"\\")

print(r"    \bottomrule")
print(r"\end{tabular}")