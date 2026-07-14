import numpy as np
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import os

import csv
from source.models import Models
from source.morph2 import Morph

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import time
import warnings

warnings.filterwarnings('ignore')

# 'ECG200', 'Epilepsy', 'ECGFiveDays', 'StandWalkJump', 'TwoLeadECG','NerveDamage' ,'MedicalImages', 'Colposcopy', 'EyesOpenShut', 'ToeSegmentation1'

ECG_datasets = ['MedicalImages']
    
all_results = {}
# 123, 0, 42, 2839,
seed_list = [3291]


# Ensure output directories exist
os.makedirs('Expansion/final/linear', exist_ok=True)
os.makedirs('Expansion/final/linear/std', exist_ok=True)
os.makedirs('Expansion/final/linear/results', exist_ok=True)


for df_name in ECG_datasets:
    all_results[df_name] = {}
    dataset_rows = [] 
    
    csv_file_path = f'Expansion/final/{df_name}.csv'

    for seed in seed_list:    
        try:
            # Load Dataset ===================================
            X, y = load_classification(df_name)
            le = LabelEncoder()
            y = le.fit_transform(y)
        except Exception as e:
            print(f'{df_name}: Dataset Not Available - {str(e)}')
            continue

        print("-" * 70)
        print(f"Dataset Name: {df_name} | Seed: {seed}")
        print("-" * 50)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Useful Information
        ts_length = X_train.shape[2]
        df_size = X_train.shape[0]
        n_classes = len(np.unique(y))  
        variates = X_train.shape[1]  
        class_counts = Counter(y_train)

        print('X_train Shape:', X_train.shape)
        print('X_test Shape:', X_test.shape)
        print("-" * 50)

        # Train and Evaluate Models ===================================
        lstm = Models('lstm', X_train, y_train, seed=seed)
        lstm.train_lstm()
        catch = Models('catch22', X_train, y_train, seed=seed)
        catch.train_catch22()
        rocket = Models('rocket', X_train, y_train, seed=seed)
        rocket.train_rocket()
        inception = Models('inception', X_train, y_train, seed=seed)
        inception.train_inception()

        models = [lstm, catch, rocket, inception] 
        acc = {}
        for m in models:
            pred, _ = m.predict(X_test)
            acc[m.model_name] = accuracy_score(pred, y_test)

        print(acc)
        print("-" * 50)

        # Loop Through each Class ===================================
        for c in np.unique(y):
            start_class = time.time()
            print(f'Processing Class: {c}')

            class_perc = round(class_counts[c] / df_size, 3)

            # Compute Morphing ===================================
            morphing = Morph(X_test, y_test, c, use_dba=False)
            morphing.get_DTWGlobalBorderline(X_test.shape[0])
            results = morphing.CalculateMorph(models)

            end_class = time.time()
            print(f'Total Class {c} run time: {end_class - start_class}')

            class_decoded = le.inverse_transform([c])[0]
            
            if class_decoded not in all_results[df_name]:
                all_results[df_name][class_decoded] = {}
            all_results[df_name][class_decoded][seed] = results

            # Append raw metrics to dataset-specific collection
            for model in results.keys():
                data = results[model]['metrics']
                dataset_rows.append({
                    'dataset': df_name,
                    'df_size': df_size,
                    'n_variates': variates,
                    'ts_length': ts_length,
                    'n_classes': n_classes,
                    'class': class_decoded,
                    'class_perc': class_perc,
                    'model': model,
                    'seed': seed,
                    'metric_mean': float(data['mean']),
                    'metric_std': float(data['std']),
                    'model_acc': acc[model]
                })
            
            # save  dataset_rows to a txt file
            with open(f'Expansion/final/' + f'{df_name}_{seed}.txt', 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=dataset_rows[0].keys())
                writer.writeheader()
                writer.writerows(dataset_rows)

        # Clean ups per seed
        del models, lstm, catch, rocket, X_train, X_test, y_train, y_test

    # =====================================================================
    # DATASET AGGREGATION & SAVING (Triggered at the end of EACH dataset)
    # =====================================================================
    if dataset_rows:
        print(f"\nAggregating and saving results for {df_name}...")
        df_raw = pd.DataFrame(dataset_rows)

        groupby_cols = ['dataset', 'df_size', 'n_variates', 'ts_length', 'n_classes', 'class', 'class_perc', 'model']
        
        df_aggregated = df_raw.groupby(groupby_cols).agg(
            mean_of_metric_mean=('metric_mean', 'mean'),
            std_of_metric_mean=('metric_mean', 'std'),
            mean_of_metric_std=('metric_std', 'mean'),
            std_of_metric_std=('metric_std', 'std'),
            mean_model_acc=('model_acc', 'mean'),
            std_model_acc=('model_acc', 'std')
        ).reset_index()

        # Append to CSV: Write headers only if the file doesn't exist yet
        file_exists = os.path.isfile(csv_file_path)
        df_aggregated.to_csv(csv_file_path, mode='a', index=False, header=not file_exists)

        # Save individual dataset raw pickle file (retains original behavior)
        with open(f'Expansion/final/linear/std/{df_name}.pkl', 'wb') as f:
            pickle.dump(all_results[df_name], f)

        # Save cumulative rolling pickle of all datasets processed so far
        with open('Expansion/final/linear/all_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)

        print(f"Finished saving {df_name}.\n")

print('All datasets processed, aggregated, and saved successfully!')