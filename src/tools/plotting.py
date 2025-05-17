import os
import matplotlib.pyplot as plt

def plot_bev(state_out, pred_out, y_true, batch_index, save_dir="result/hybridtrack/online/BEV_figures"):
    os.makedirs(save_dir, exist_ok=True)
    if state_out.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: state_out {state_out.shape} vs y_true {y_true.shape}")
    if len(state_out.shape) != 2 or state_out.shape[0] != 7:
        raise ValueError(f"Expected shape [7, seq_len], got {state_out.shape}")
    seq_len = state_out.shape[1]
    fig, ax = plt.subplots(figsize=(10, 10))
    for t in range(seq_len):
        ax.plot(state_out[0, t], state_out[2, t], 'bo', label='State' if t == 0 else "")
        ax.plot(pred_out[0, t], pred_out[2, t], 'go', label='Predicted' if t == 0 else "")
        ax.plot(y_true[0, t], y_true[2, t], 'ro', label='Ground Truth' if t == 0 else "")
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(f'BEV for Batch {batch_index}')
    ax.legend()
    ax.grid(True)
    plt.savefig(os.path.join(save_dir, f'bev_batch_{batch_index}.png'))
    plt.close(fig)

def plot_all_batches(state_out_batches, pred_out_batches, y_true_batches):
    batch_size = state_out_batches.shape[0]
    for i in range(batch_size):
        plot_bev(state_out_batches[i], pred_out_batches[i], y_true_batches[i], i)
