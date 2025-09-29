import math
import matplotlib.pyplot as plt

def plot_batch_pred_vs_actual(pred_tensor, actual_tensor, n=4, conf_threshold=0.1,
                              match_indices=None, show_confidence=True):
    """
    Plot side-by-side subplots of predicted vs actual points.

    Parameters:
    - pred_tensor: (B, M, 3) — predicted [x, y, confidence]
    - actual_tensor: (B, M, 3) — ground truth [x, y, confidence]
    - n: int — number of batch elements to plot
    - conf_threshold: float — confidence below which predictions are hidden
    - match_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]]
        Hungarian match results for each batch element (row_idx, col_idx).
        If provided, matched predictions are highlighted differently.
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
        ax.scatter(true_coords[:, 0], true_coords[:, 1],
                   c='blue', label='Actual', alpha=0.6)

        # Filter out low-confidence predictions
        keep_mask = pred_conf > conf_threshold
        kept_coords = pred_coords[keep_mask]
        kept_conf = pred_conf[keep_mask]

        if match_indices is not None:
            row_idx, _ = match_indices[i]
            matched_mask = keep_mask.copy()
            matched_mask[:] = False
            matched_mask[row_idx] = True

            # Plot matched predictions in solid red
            ax.scatter(pred_coords[matched_mask, 0], pred_coords[matched_mask, 1],
                       c='red', alpha=kept_conf[matched_mask] if show_confidence else 1.0,
                       label='Matched')
            
            # Plot unmatched predictions in gray
            unmatched_mask = keep_mask & ~matched_mask
            ax.scatter(pred_coords[unmatched_mask, 0], pred_coords[unmatched_mask, 1],
                       c='gray', alpha=kept_conf[unmatched_mask] if show_confidence else 0.5,
                       label='Unmatched')
        else:
            # Just plot all kept predictions normally
            ax.scatter(kept_coords[:, 0], kept_coords[:, 1],
                       c='red', alpha=kept_conf if show_confidence else 1.0,
                       label='Predicted')

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
