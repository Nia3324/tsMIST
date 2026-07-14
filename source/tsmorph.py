import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        Perform linear or path-interpolation morphing for multivariate time series.

        Returns:
            np.array: Morphed time series (shape: [n, d, t])
                      where n = granularity steps, d = dimensions, t = time_points
        """
        min_length = min(self.S.shape[1], self.T.shape[1])
        self.S = self.S[:, -min_length:].astype(float)
        self.T = self.T[:, -min_length:].astype(float)

        if use_dba:
            # Apply PATH-INTERPOLATION per dimension.
            n_dims = self.S.shape[0]
            morphed_dims = []
            
            # Create alpha weights, excluding exactly 0.0 (S) and 1.0 (T)
            # self.granularity already includes the +2 from __init__
            alpha = np.linspace(0, 1, self.granularity)[1:-1]

            for dim in range(n_dims):
                s_1d = self.S[dim, :]
                t_1d = self.T[dim, :]
                
                # _path_interp returns shape: (time, steps)
                # We transpose it to (steps, time) for stacking
                dim_steps = self._path_interp(s_1d, t_1d, alpha).T
                
                # Store steps for this dimension
                morphed_dims.append(dim_steps)

            # Stack dimensions -> Shape: (dimensions, granularity_steps, time_points)
            data = np.stack(morphed_dims)
            
            # Swap axes to return format: (granularity_steps, dimensions, time_points)
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
        return plt


    def _path_interp(self, S: np.ndarray, T: np.ndarray, alphas: np.ndarray) -> np.ndarray:
        """Path-interpolation morph.

        Computes the DTW correspondence between S and T once. For each matched pair
        (i, j) and morphing weight a, a sample is placed at the interpolated time
        (1 - a) * i + a * j with the interpolated amplitude (1 - a) * S[i] + a * T[j];
        the scattered samples are then resampled onto the regular integer grid.

        Returns an array of shape (len(S), len(alphas)) = (time, steps).
        """
        path = self._dtw_path(S, T)
        I = np.array([i for i, _ in path], dtype=float)
        J = np.array([j for _, j in path], dtype=float)
        Sv = S[I.astype(int)]
        Tv = T[J.astype(int)]
        L = len(S)
        grid = np.arange(L, dtype=float)

        columns = []
        for a in alphas:
            t = (1.0 - a) * I + a * J                 # interpolated time positions
            v = (1.0 - a) * Sv + a * Tv               # interpolated amplitudes
            order = np.argsort(t, kind="stable")
            ts, vs = t[order], v[order]
            # average samples that land on the same interpolated time
            ut, inv = np.unique(ts, return_inverse=True)
            uv = np.zeros(len(ut))
            cnt = np.zeros(len(ut))
            np.add.at(uv, inv, vs)
            np.add.at(cnt, inv, 1.0)
            uv /= cnt
            columns.append(np.interp(grid, ut, uv))   # resample onto integer grid

        return np.vstack(columns).T                   # (time, steps)

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
        """DBA implementation for a list of 1D sequences.
        Note: Kept for legacy compatibility. Use _path_interp instead.
        """
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
        Note: Kept for legacy compatibility. Use _path_interp instead.
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
        warped[~mask] = centroid[~mask]
        return warped