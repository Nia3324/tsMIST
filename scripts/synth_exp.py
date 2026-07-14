from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from source.models import Models
from source.morph2 import Morph
from source.generation import Generation


shift_values = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
results = {}
for s in shift_values:
    # Generate dataset with shift
    gen = Generation(base_functions=('sin', 'sin'), n_samples=500, frequencies=(0.5,0.5), noise_level=(0.1, 0.1), shift_vert=s, shif_horz=s) 
    X, y = gen.generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    lstm = Models('lstm', X_train, y_train.ravel())
    lstm.train_lstm()
    catch22 = Models('catch22', X_train, y_train)
    catch22.train_catch22()
    rocket = Models('rocket', X_train, y_train)
    rocket.train_rocket()
    inception = Models('inception', X_train, y_train)
    inception.train_inception()

    # Calculate morphing
    models = [catch22, catch22, rocket, inception]
    res = {}
    morph = Morph(X_test, y_test, use_dba=True)
    morph.get_DTWGlobalBorderline(X_test.shape[0]) 
    res = morph.CalculateMorph(models, )
    results[s] = res
   
columns = ['n_pairs', 'model', 'mean', 'std']

data = []
for s in results.keys():
    for model in results[s].keys():
        metrics = results[s][model]['metrics']
        line = [s, model, metrics['mean'], metrics['std']]
        data.append(line)

# save to csv
df = pd.DataFrame(data, columns=columns)
df.to_csv('final/pathi_synth.csv', index=False)
