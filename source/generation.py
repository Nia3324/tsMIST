import numpy as np
import pandas as pd
from scipy import signal

class Generation:
    def __init__(self, base_functions = ('sin', 'sin'), n_samples = 200, class_ratio = 0.5,
                 n_timepoints = 100, frequencies = (0.5, 0.5), noise_level = (0.1, 0.1), shift_vert = 0.1, shif_horz = 0.1):
        
        self.base_functions = base_functions
        self.n_samples = n_samples
        self.class_ratio = class_ratio
        self.n_timepoints = n_timepoints
        self.frequencies = frequencies
        self.noise_level = noise_level
        self.shift_vert = shift_vert
        self.shift_horz = shif_horz

        self.class0 = None
        self.class1 = None
        self.X = None
        self.y = None
        return

    def generate_class(self, func : str, freq : int, 
                       samples : int, noise_lvl : int, apply_sift=False):
        # Generate time points
        t = np.linspace(0, 10, self.n_timepoints)
        
        # Initialize DataFrame
        df = pd.DataFrame()
        df['time'] = t
        
        # Generate base signals
        base_signals = {}
        
        if func == 'sin':
            if apply_sift:
                signal_data = np.sin(2 * np.pi * freq * t + self.shift_horz) 
            else:
                signal_data = np.sin(2 * np.pi * freq * t) 
        elif func == 'cos':
            if apply_sift:
                signal_data = np.cos(2 * np.pi * freq * t + self.shift_horz) 
            else:
                signal_data = np.cos(2 * np.pi * freq * t) 
        elif func == 'sawtooth':
            if apply_sift:
                signal_data = signal.sawtooth(2 * np.pi * freq * t + self.shift_horz)  
            else:
                signal_data = signal.sawtooth(2 * np.pi * freq * t) 
        elif func == 'square':
            if apply_sift:
                signal_data = signal.square(2 * np.pi * freq * t + self.shift_horz) 
            else:
                signal_data = signal.square(2 * np.pi * freq * t) 
                
        base_signals[f'{func}_{freq}'] = signal_data
        
        signals_array = np.zeros((self.n_timepoints, samples))
        for i in range(samples):
            weights = np.random.uniform(-1, 1, len(base_signals))
            combined_signal = np.zeros_like(t)
            
            for (name, sig), weight in zip(base_signals.items(), weights):
                combined_signal += weight * sig
            
            # Normalize
            combined_signal = combined_signal / np.max(np.abs(combined_signal))
            
            # Add random noise
            noise = np.random.normal(0, noise_lvl, self.n_timepoints)
            if apply_sift:
                final_signal = abs(combined_signal) + noise + self.shift_vert
            else: 
                final_signal = abs(combined_signal) + noise

            signals_array[:, i] = final_signal

        df = pd.DataFrame(signals_array, columns=[str(i+1) for i in range(samples)])    
        return df
    
    def generate_data(self) -> None:
        # Generate class 0
        self.class0 = self.generate_class(self.base_functions[0], self.frequencies[0], 
                                          int(self.n_samples * self.class_ratio), 
                                          self.noise_level[0], apply_sift=True)
        
        # Generate class 1
        self.class1 = self.generate_class(self.base_functions[1], self.frequencies[1], 
                                          self.n_samples - int(self.n_samples * self.class_ratio), 
                                          self.noise_level[1], apply_sift=False)
        
        X_class0 = self.class0.T
        X_class1 = self.class1.T
        
        X_df = pd.concat([X_class0, X_class1])
        self.X = X_df.values
        self.X = np.expand_dims(self.X, axis=1)      

        # Create labels
        y_class0 = np.zeros(X_class0.shape[0])
        y_class1 = np.ones(X_class1.shape[0])
        self.y = np.concatenate([y_class0, y_class1])
         
        # Shuffle the data
        # shuffle_idx = np.random.permutation(len(self.y))
        # self.X = self.X[shuffle_idx]
        # self.y = self.y[shuffle_idx]

        return self.X, self.y