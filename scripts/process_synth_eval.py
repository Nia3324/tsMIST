import pandas as pd

# 1. Load the data
df = pd.read_csv('final/synthEval.csv')

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
    'DTW_DBA', 'Spectral_DBA',
    'DTW_Linear', 'Spectral_Linear',
]

# 4. Group and aggregate
grouped = df.groupby(['dataset'])[cols_to_agg].mean()

# 5. Save the aggregated data to CSV
# 'reset_index()' makes dataset and model regular columns instead of indices
grouped.reset_index().to_csv('final/aggregated_synth_eval.csv', index=False)
print("Aggregated data saved to 'aggregated_synth_eval.csv'")



