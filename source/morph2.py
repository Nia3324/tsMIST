from matplotlib import use
import numpy as np
from aeon.distances import dtw_distance
from typing import Dict, Tuple
from numba import njit, prange
from source.models import Models
from source.tsmorph import TSmorph
from tqdm import tqdm

class Morph:
    def __init__(self, X: np.ndarray, y: np.ndarray, target_class: int = 1, use_dba: bool = False) -> None:
        """
        Initialize Morph class for morphing analysis
        
        Parameters:
        -----------
        X : np.ndarray
            Input feature array (shape: [n_samples, n_dimensions, n_timepoints])
        y : np.ndarray
            Target labels (shape: [n_samples])
        target_class : int, optional
            Class to focus morphing analysis on (default is 1)
        """
        self.X = X  # Shape: [n_samples, n_dimensions, n_timepoints]
        self.y = y
        self.target_class = target_class
        self.use_dba = use_dba
        
        self.class1_mask = self.y == target_class
        self.class0_mask = self.y != target_class

        self.class1_X = self.X[self.class1_mask]  # Shape: [n_class1, n_dimensions, n_timepoints]
        self.class1_y = self.y[self.class1_mask]
        self.class0_X = self.X[self.class0_mask]  # Shape: [n_class0, n_dimensions, n_timepoints]
        self.class0_y = self.y[self.class0_mask]
        
        self.distances = []
        self.borderline_pairs = {}


    @staticmethod
    @njit(parallel=True)
    def compute_dtw_distances(class0_X: np.ndarray, class1_X: np.ndarray) -> np.ndarray:
        """
        Compute average DTW distances between two classes for multivariate time series
        
        Parameters:
        -----------
        class0_X : np.ndarray
            Samples from class 0 (shape: [n_class0, n_dimensions, n_timepoints])
        class1_X : np.ndarray
            Samples from class 1 (shape: [n_class1, n_dimensions, n_timepoints])
        
        Returns:
        --------
        np.ndarray
            Array of average DTW distances (shape: [n_class0 * n_class1])
        """
        n0, n1 = len(class0_X), len(class1_X)
        distances = np.zeros(n0 * n1, dtype=np.float64)
        
        for i in prange(n0):
            for j in range(n1):
                # Compute DTW distance for each dimension and take the average
                avg_distance = 0.0
                for d in range(class0_X.shape[1]):  # Iterate over dimensions
                    avg_distance += dtw_distance(class0_X[i, d], class1_X[j, d])
                distances[i * n1 + j] = avg_distance / class0_X.shape[1]  # Average across dimensions
        
        return distances
    
    def get_DTWGlobalBorderline(self, size=1) -> None:
        """
        Compute global borderline pairs based on average DTW distances
        
        Parameters:
        -----------
        perc_samples : float
            Percentage of samples to consider for borderline pairs
        """
        distances = self.compute_dtw_distances(self.class0_X, self.class1_X)
        
        # Sort distances and select top pairs
        sorted_indices = np.argsort(distances)
        #n_samples = min(round(len(sorted_indices) * perc_samples), len(sorted_indices))
        n_samples = min(size, len(sorted_indices))
        # Store distances and pair indices
        self.distances = distances[sorted_indices[:n_samples]]
        
        # Reconstruct pair indices
        indices = [
            (i, j) 
            for i in range(len(self.class0_X)) 
            for j in range(len(self.class1_X))
        ]
        self.allborderline = {
            indices[idx]: distances[idx] 
            for idx in sorted_indices
        }
        self.borderline_pairs = {
            indices[idx]: distances[idx] 
            for idx in sorted_indices[:n_samples]
        }
    

    def get_AllMorphs(self, granularity:int=100, keep_original_serie: bool = True) -> np.ndarray:
        morphs = []

        for pair, _ in self.borderline_pairs.items(): 
            source_c0 = self.class0_X[pair[0]]  # Shape: [n_dimensions, n_timepoints]
            target_c1 = self.class1_X[pair[1]]  # Shape: [n_dimensions, n_timepoints]
                
            # Apply morphing
            if self.use_dba:
                morphing = TSmorph(S=source_c0, T=target_c1, granularity=granularity).transform(use_dba=True)
            else:
                morphing = TSmorph(S=source_c0, T=target_c1, granularity=granularity).transform()
            
            if not keep_original_serie:
                morphing = morphing[1:-1]  # Exclude original series

            morphs.append(morphing)

        return np.array(morphs)
    

    def CalculateMorph(self,  models: Tuple[Models], granularity:int=100) -> np.ndarray:
        morphs = []
        results = {}

        for pair, _ in self.borderline_pairs.items():
            source_c0 = self.class0_X[pair[0]]  
            target_c1 = self.class1_X[pair[1]]  
            source_c0_y = self.class0_y[pair[0]]
            target_c1_y = self.class1_y[pair[1]]

            if self.use_dba:
                m = TSmorph(S=source_c0, T=target_c1, granularity=granularity).transform(use_dba=True)
            else:
                m = TSmorph(S=source_c0, T=target_c1, granularity=granularity).transform()
            morphs.append(m)

        for model in models:
            results[model.model_name] = {}
            good_morphs = []
            good_pred = []
            indices = []
            percentage = []

            for morphing in tqdm(morphs):
                # Predict new labels using selected model
                if model.model_name == 'lstm':
                    if len(morphing.shape) != 3:  # Ensure it has 3 dimensions (samples, sequence_length, features)
                        morphing = morphing.reshape(morphing.shape[0], 1, morphing.shape[1])
                    pred,_ = model.predict(morphing)
                else:
                    pred,_ = model.predict(morphing)

                # Ensure valid morphing pairs
                if pred[0] == source_c0_y and pred[-1] == target_c1_y:
                        
                    # Find where label changes
                    change_idx = 1
                    for i in range(1, len(pred)-1):  # account for both original series
                        if pred[i] != source_c0_y:
                            change_idx = i
                            break

                    # Calculate morphing percentage 
                    perc = 1/granularity * change_idx
                                    
                    good_morphs.append(morphing)
                    good_pred.append(round(perc, 2))
                    indices.append(change_idx)
                    percentage.append(perc)
          
         
            results[model.model_name]['morphs'] = good_morphs
            results[model.model_name]['model_preds'] = good_pred
            results[model.model_name]['change_perc'] = percentage 
            results[model.model_name]['change_indice'] = indices 
                        
        # Compute metrics for each model
        for _, model_results in results.items():
            morphs_perc = model_results['change_perc']
            model_results['metrics'] = {
                'mean': float(np.mean(morphs_perc)) if morphs_perc else 0.0,
                'std': float(np.std(morphs_perc)) if morphs_perc else 0.0
            }
        return results


    def Binary_MorphingCalculater(
        self, 
        models: Tuple[Models], 
        granularity: int = 100, 
        verbose: bool = False
    ) -> Dict:
        results = {}
        
        for pair, _ in tqdm(self.borderline_pairs.items()):
            good_morphs = []
            source_c0 = self.class0_X[pair[0]]  # Shape: [n_dimensions, n_timepoints]
            source_c0_y = self.class0_y[pair[0]]
            target_c1 = self.class1_X[pair[1]]  # Shape: [n_dimensions, n_timepoints]
            target_c1_y = self.class1_y[pair[1]]
            
            # Apply morphing
            morphing = TSmorph(S=source_c0, T=target_c1, granularity=granularity).transform()
            
            for model in models:
                # Initialize model-specific results if not exists
                if model.model_name not in results:
                    results[model.model_name] = {
                        'morphs': {},
                        'model_preds': {},
                        'pair_results': {},
                        'morphs_perc': []
                    }
                
                if model.model_name:
                    # Predict new labels using selected model
                    if model.model_name == 'lstm':
                        if len(morphing.shape) != 3:  # Ensure it has 3 dimensions (samples, sequence_length, features)
                            morphing = morphing.reshape(morphing.shape[0], 1, morphing.shape[1])
                        pred,_ = model.predict(morphing)
                else:
                    pred,_ = model.predict(morphing)

                # Ensure valid morphing pairs
                if pred[0] == source_c0_y and pred[-1] == target_c1_y:
                    
                    # Find where label changes
                    change_idx = 1
                    for i in range(1, len(pred)-1):  # account for both original series
                        if pred[i] != source_c0_y:
                            change_idx = i
                            break
            
                    # Calculate morphing percentage 
                    perc = 1/granularity * change_idx
                    
                    results[model.model_name]['morphs'][pair] = morphing
                    results[model.model_name]['model_preds'][pair] = pred
                    results[model.model_name]['pair_results'][pair] = round(perc, 2)
                    results[model.model_name]['morphs_perc'].append(perc)
                    
                    if verbose:
                        print(f"Model {model} Pair: {pair} -> Morphing percentage: {perc:.2f}")

        # Compute metrics for each model
        for model_name, model_results in results.items():
            morphs_perc = model_results['morphs_perc']
            model_results['metrics'] = {
                'mean': float(np.mean(morphs_perc)) if morphs_perc else 0.0,
                'std': float(np.std(morphs_perc)) if morphs_perc else 0.0
            }
        
        return results
