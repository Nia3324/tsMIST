import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Assuming these exist in your .utils, otherwise you can comment them out
# from .utils import plot_gradient_timeseries, nmae

class TSmorph:
    def __init__(self, S: np.array, T: np.array, granularity: int) -> None:
        """
        Args:
            S (np.array): Source time series (shape: [dimensions, time_points] or [time_points])
            T (np.array): Target time series (shape: [dimensions, time_points] or [time_points])
            granularity (int): Number of steps in the morphing process
        """
        # FIX 1: Handle 1D inputs by reshaping to (1, time_points)
        self.S = S if S.ndim == 2 else S.reshape(1, -1)
        self.T = T if T.ndim == 2 else T.reshape(1, -1)
        
        # We add 2 to granularity to account for the start (S) and end (T) in the grid
        self.granularity = granularity + 2

    def transform(self, use_dba: bool = False) -> np.array:
        """
        Perform linear morphing for multivariate time series.

        Returns:
            np.array: Morphed time series (shape: [n, d, t])
                      where n = granularity, d = dimensions, t = time_points
        """
        min_length = min(self.S.shape[1], self.T.shape[1])
        self.S = self.S[:, -min_length:].astype(float)
        self.T = self.T[:, -min_length:].astype(float)

        if use_dba:
            # FIX 2: Apply DBA per dimension. 
            # _dba expects 1D arrays, so we loop over the dimensions (d).
            n_dims = self.S.shape[0]
            morphed_dims = []

            for dim in range(n_dims):
                s_1d = self.S[dim, :]
                t_1d = self.T[dim, :]
                
                # Compute centroid and aligned versions for this dimension
                centroid = self._dba([s_1d, t_1d])
                S_aligned = self._warp_to_centroid(s_1d, centroid)
                T_aligned = self._warp_to_centroid(t_1d, centroid)

                # Interpolate between aligned Source and Target
                # We skip the very first (0.0) and very last (1.0) because 
                # those are just S and T, unless you want them included in the morph set.
                # The original code logic suggests we want 'self.granularity' items in total.
                alpha = np.linspace(0, 1, self.granularity + 2)[1:-1]
                
                dim_steps = []
                for i in alpha:
                    dim_steps.append(i * T_aligned + (1 - i) * S_aligned)
                
                # Stack steps for this dimension -> Shape: (granularity, time_points)
                morphed_dims.append(np.vstack(dim_steps))

            # Stack dimensions -> Shape: (dimensions, granularity, time_points)
            data = np.stack(morphed_dims)
            
            # Swap axes to return format: (granularity, dimensions, time_points)
            return data.swapaxes(0, 1)

        else:
            S_aligned = self.S
            T_aligned = self.T
                
            # Morphing process (vectorized)
            # Shape: [granularity, 1, 1]
            alpha = np.linspace(0, 1, self.granularity).reshape(-1, 1, 1)
            
            # Shape: [granularity, dimensions, time_points]
            morphed_series = alpha * T_aligned + (1 - alpha) * S_aligned  
            return morphed_series

    def plot_morphed_series(self, morphed_series: np.array, 
                            start_color='#61E6AA', end_color='#5722B1', 
                            title=True, morph_labels=True) -> None:
        """
        Plot the morphed time series.
        """
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

        start_rgb = hex_to_rgb(start_color)
        end_rgb = hex_to_rgb(end_color)
        n_series = self.granularity

        colors = []
        for i in range(n_series):
            ratio = i / (n_series - 1) if n_series > 1 else 0
            r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
            g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
            b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio
            colors.append((r, g, b))

        dimensions = self.S.shape[0]
        fig, axes = plt.subplots(dimensions, 1, figsize=(12, 4 * dimensions), squeeze=False)

        for dim in range(dimensions):
            ax = axes[dim, 0]

            # Plot original source
            ax.plot(self.S[dim, :], color=start_color, linewidth=3, label='Source Series (S)')

            # Plot intermediate morphs
            # Note: morphed_series shape is (granularity, dim, time)
            # We iterate 1 to N-1 to skip overwriting S and T if they are at indices 0 and -1
            for i in range(1, self.granularity - 1):
                if i >= len(morphed_series): break
                
                source_pct = round((self.granularity - i - 1) / (self.granularity - 1) * 100)
                target_pct = round(i / (self.granularity - 1) * 100)
                
                label = f"S {source_pct}/{target_pct} T" if morph_labels else None
                ax.plot(morphed_series[i, dim, :], color=colors[i], linewidth=2, label=label)

            # Plot target
            ax.plot(self.T[dim, :], color=end_color, linewidth=3, label='Target Series (T)')

            if title:
                ax.set_title(f'Dimension {dim}', fontsize=16, pad=15)
            ax.set_xlabel('Time', fontsize=16)
            ax.set_ylabel('Value', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

        plt.tight_layout()
        plt.show()

    def _dtw_path(self, a: np.ndarray, b: np.ndarray):
        """Computes the DTW alignment path between sequences a and b."""
        n = len(a)
        m = len(b)
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist = (a[i - 1] - b[j - 1]) ** 2
                cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

        i, j = n, m
        path = []
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                path.append((i - 1, j - 1))
                prevs = [(cost[i - 1, j - 1], i - 1, j - 1), (cost[i - 1, j], i - 1, j), (cost[i, j - 1], i, j - 1)]
                _, i, j = min(prevs, key=lambda x: x[0])
            elif i > 0:
                path.append((i - 1, 0))
                i -= 1
            else:
                path.append((0, j - 1))
                j -= 1
        path.reverse()
        return path

    def _dba(self, sequences: list, n_iter: int = 10):
        """DBA implementation for a list of 1D sequences."""
        L = len(sequences[0])
        centroid = np.mean(np.vstack(sequences), axis=0)

        for _ in range(n_iter):
            accum = np.zeros(L)
            counts = np.zeros(L)
            for s in sequences:
                path = self._dtw_path(centroid, s)
                for i, j in path:
                    accum[i] += s[j]
                    counts[i] += 1
            mask = counts > 0
            centroid[mask] = accum[mask] / counts[mask]
        return centroid


    def _warp_to_centroid(self, seq: np.ndarray, centroid: np.ndarray):
        """Projects `seq` onto the indices of `centroid` using the DTW path.

        Returns an aligned sequence with the same length as `centroid`.
        """
        L = len(centroid)
        warped = np.zeros(L)
        counts = np.zeros(L)
        path = self._dtw_path(centroid, seq)
        for i, j in path:
            warped[i] += seq[j]
            counts[i] += 1
        mask = counts > 0
        warped[mask] = warped[mask] / counts[mask]
        # fallback to centroid values where nothing was mapped
        warped[~mask] = centroid[~mask]
        return warped