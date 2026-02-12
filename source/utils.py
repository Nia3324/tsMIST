import matplotlib.pyplot as plt
import numpy as np

def plot_gradient_timeseries(df, start_color='#61E6AA', end_color='#5722B1'):
    # Convert hex to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

    # Generate gradient colors
    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)
    n_series = len(df.columns)

    colors = []
    for i in range(n_series):
        ratio = i / (n_series - 1)
        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio
        colors.append((r, g, b))

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot each series with its corresponding color
    for idx, column in enumerate(df.columns):
        plt.plot(df.index, df[column], color=colors[idx], label=column, linewidth=2)

    # Customize the plot
    plt.title('Morphed Time Series', fontsize=14, pad=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()
    return

def nmae(y, y_hat):
    """
    Computes the Normalized Mean Absolute Error (NMAE).

    Args:
        y (np.array): True values.
        y_hat (np.array): Predicted values.
    
    Returns:
        float: NMAE score.
    """
    return np.mean(np.abs(y - y_hat)) / np.mean(np.abs(y))