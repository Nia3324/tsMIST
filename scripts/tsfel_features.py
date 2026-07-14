import numpy as np
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

import csv

from source.models import Models
from source.morph2 import Morph

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import time
import pickle
import warnings
import tsfel
warnings.filterwarnings('ignore')


ECG_datasets = ['ECG200']


def multivariate_featues(row, cfg): 
    feature_names = []  
    row_features = []  

    for var_idx, variate in enumerate(row):
        features = tsfel.time_series_features_extractor(cfg, variate, verbose=0)
        
        modified_columns = [
            col.replace('0_', f'{var_idx}_') if var_idx != 0 else col 
            for col in features.columns
        ]
        features.columns = modified_columns  

        row_features.append(features)
        feature_names.extend(modified_columns) 
    
    combined_row_features = pd.concat(row_features, axis=1)
    dataframe = pd.DataFrame(combined_row_features, columns=feature_names)
    return dataframe

    
all_results = {}
all_features = {}
results_array = np.empty((0, 11)) 

for df_name in ECG_datasets:
    try:
        # Load Dataset ===================================
        X, y = load_classification(df_name)
        le = LabelEncoder()
        y = le.fit_transform(y)
    except Exception as e:
        print(f'{df_name}: Dataset Not Available - {str(e)}')
        continue

    print("-" * 70)
    print("Dataset Name:", df_name)
    print("-" * 50)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Useful Information
    ts_length = X_train.shape[2]
    df_size = X_train.shape[0]
    n_classes = len(np.unique(y))  
    variates = X_train.shape[1]  
    class_counts = Counter(y_train)
    num_classes = len(class_counts)

    print('X_train Shape:', X_train.shape)
    print('X_test Shape:', X_test.shape)

    print('y_train Counts:', np.unique(y_train, return_counts=True))
    print('y_test Counts:', np.unique(y_test, return_counts=True))
    print("-" * 50)


    # Train and Evaluate Models ===================================
    #start = time.time()
    #lstm = Models('lstm', X_train, y_train)
    #lstm.train_lstm()
    #catch = Models('catch22', X_train, y_train)
    #catch.train_catch22()
    #rocket = Models('rocket', X_train, y_train)
    #rocket.train_rocket()
    inception = Models('inception', X_train, y_train)
    inception.train_inception()

    models = [inception] # [lstm, catch, rocket,
    acc = {}
    for m in models:
        pred, _ = m.predict(X_test)
        acc[m.model_name] = accuracy_score(pred, y_test)

    print(acc)
    print("-" * 50)


    class_results = {}
    features_results = {}
    # Loop Through each Class ===================================
    for c in np.unique(y):
        start_class = time.time()
        print(f'Processing Class: {c}')

        # Calculate class percentage
        class_perc = round(class_counts[c] / df_size, 3)

        # Compute Morphing ===================================
        morphing = Morph(X_test, y_test, c, use_dba=True)
        morphing.get_DTWGlobalBorderline(X_test.shape[0])
        results = morphing.CalculateMorph(models)

        end_class = time.time()
        print(f'Total Class {c} run time: {end_class - start_class}')

        class_decoed = le.inverse_transform([c])[0]
        print(class_decoed)
        class_results[class_decoed] = results


        # Feature Extraction using TSFEL ===================================
        # normalize features difference 
        models_features = {} 
        feature_names = None
        cfg_file = tsfel.get_features_by_domain() 
        for m in models:
            differences = []
            data = results[m.model_name]
            for i in range(len(data['morphs'])):

                morph = data['morphs'][i]
                change_inx = data['change_indice'][i]
               
                source = morph[0]
                target = morph[change_inx]
                
                # Univariate Time Series ===================================
                if morph.shape[1] == 1:
                    source_features = tsfel.time_series_features_extractor(cfg_file, source[0], verbose=0)
                    target_features = tsfel.time_series_features_extractor(cfg_file, target[0], verbose=0)
                    #print(source.shape, target.shape)

                # Multivariate Time Series ===================================
                else:
                    source_features = multivariate_featues(source, cfg_file)
                    target_features = multivariate_featues(target, cfg_file)

                if feature_names is None:
                    feature_names = source_features.columns
                
                source_features = np.array(source_features)
                target_features = np.array(target_features)

                percentage_diff = (target_features - source_features) / (np.abs(source_features) + 1e-10)

                differences.append(percentage_diff)

            avg_diff = np.mean(differences, axis=0).flatten()            
                        
            if np.isnan(avg_diff[0]):
                print(f"avg_diff is None for {m.model_name} in {class_decoed}. No correct model predictions!")
                models_features[m.model_name] = None
                continue
            else:
                diffs = {}
                for i in range(len(avg_diff)):
                    diffs[feature_names[i]] = avg_diff[i]

                models_features[m.model_name]  = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)

                print("-" * 50)
                print('Top 5 Feature Changes:')
                for feature, diff in models_features[m.model_name][:5]:
                    print(f'{feature}: {diff}')

        features_results[class_decoed] = models_features
     
        # Append results to NumPy array
        for model in results.keys():
            data = results[model]['metrics']
            line = np.array([[df_name, df_size, variates, ts_length, n_classes, class_decoed, class_perc, model, data['mean'], data['std'], acc[model]]])
            results_array = np.vstack((results_array, line))
        

    # Save results for the current dataset
    file_name = f'Expansion/{df_name}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(class_results, f)

    # Save features results for the current dataset
    file_name = f'Expansion/features_{df_name}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(features_results, f)

    # Add results to all_results
    all_results[df_name] = class_results
    all_features[df_name] = features_results

    # Convert NumPy array to Pandas DataFrame
    columns = ['dataset', 'df_size', 'n_variates','ts_length', 'n_classes', 'class', 'class_perc', 'model', 'mean', 'std', 'model_acc']
    dataframe = pd.DataFrame(results_array, columns=columns)
    # Save results to CSV
    dataframe.to_csv('Expansion/final_results2.csv', index=False)

    # Save all results to a single pickle file
    with open('Expansion/final_results2.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Save all features to a single pickle file
    with open('Expansion/final_features2.pkl', 'wb') as f:
        pickle.dump(all_features, f)

print('All results saved successfully!')
