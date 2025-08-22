import math
import matplotlib.pyplot as plt

def plot_batch_pred_vs_actual(pred_tensor, actual_tensor, n=4, show_confidence=True, threshold=0.05):
    """
    Plot side-by-side subplots of predicted vs actual points.

    Parameters:
    - pred_tensor: (B, M, 3) — predicted [x, y, confidence]
    - actual_tensor: (B, M, 3) — ground truth [x, y, confidence]
    - n: int — number of batch elements to plot
    - show_confidence: bool — whether to use confidence as alpha for predicted
    """
    pred_tensor = pred_tensor.detach().cpu()
    actual_tensor = actual_tensor.detach().cpu()

    batch_size = pred_tensor.shape[0]
    n = min(n, batch_size)

    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(n):
        pred_coords = pred_tensor[i, :, :2].numpy()
        pred_conf = pred_tensor[i, :, 2].numpy()

        true_coords = actual_tensor[i, :, :2].numpy()
        true_conf = actual_tensor[i, :, 2].numpy()
        true_coords = true_coords[true_conf > 0]

        ax = axes[i]
        ax.scatter(true_coords[:, 0], true_coords[:, 1], c='blue', label='Actual', alpha=0.6)
        if show_confidence:
            ax.scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', alpha=pred_conf.clip(threshold, 1.0), label='Predicted')
        else:
            ax.scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', label='Predicted')

        ax.set_title(f'Example {i}')
        ax.axis('equal')
        ax.grid(True)

    # Turn off unused axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()
    plt.savefig("pred_vs_actual.png")
    plt.clf()
