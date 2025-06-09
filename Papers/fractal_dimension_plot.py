import numpy as np
import matplotlib.pyplot as plt

def plot_box_counting(data, box_sizes):
    """Plot log(N) vs log(1/box size) per il box counting."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1,1)
    N_dim = data.shape[1]
    counts = []
    for size in box_sizes:
        bins = [np.arange(np.min(data[:, dim]), np.max(data[:, dim]) + size, size) for dim in range(N_dim)]
        hist, _ = np.histogramdd(data, bins=bins)
        counts.append(np.sum(hist > 0))
    log_inv_box_sizes = np.log(1/np.array(box_sizes))
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_inv_box_sizes, log_counts, 1)
    plt.figure()
    plt.plot(log_inv_box_sizes, log_counts, 'o-', label='Data')
    plt.plot(log_inv_box_sizes, np.polyval(coeffs, log_inv_box_sizes), '--', label=f'Fit: slope={coeffs[0]:.3f}')
    plt.xlabel('log(1/box size)')
    plt.ylabel('log(N box)')
    plt.legend()
    plt.title('Box-counting plot')
    plt.show()

def plot_correlation_dimension(data, r_vals):
    """Plot log(C(r)) vs log(r) per la correlation dimension."""
    data = np.asarray(data)
    dists = np.linalg.norm(data[:, np.newaxis] - data[np.newaxis, :], axis=-1)
    N = len(data)
    C = []
    for r in r_vals:
        C.append(np.sum(dists < r) / (N * (N - 1)))
    log_r = np.log(r_vals)
    log_C = np.log(C)
    coeffs = np.polyfit(log_r, log_C, 1)
    plt.figure()
    plt.plot(log_r, log_C, 'o-', label='Data')
    plt.plot(log_r, np.polyval(coeffs, log_r), '--', label=f'Fit: slope={coeffs[0]:.3f}')
    plt.xlabel('log(r)')
    plt.ylabel('log(C(r))')
    plt.legend()
    plt.title('Correlation dimension plot')
    plt.show()

def plot_multifractal(data, box_sizes, q_list):
    """Plot multifractal scaling: log(∑p^q) vs log(box size) for several q."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1,1)
    N_dim = data.shape[1]
    plt.figure()
    for q in q_list:
        S_q = []
        for size in box_sizes:
            bins = [np.arange(np.min(data[:, dim]), np.max(data[:, dim]) + size, size) for dim in range(N_dim)]
            hist, _ = np.histogramdd(data, bins=bins)
            p = hist.flatten() / np.sum(hist)
            p = p[p > 0]
            if q == 1:
                S = -np.sum(p * np.log(p))
            else:
                S = np.sum(p ** q)
            S_q.append(S)
        log_size = np.log(box_sizes)
        log_Sq = np.log(S_q)
        plt.plot(log_size, log_Sq, 'o-', label=f'q={q}')
    plt.xlabel('log(box size)')
    plt.ylabel('log(∑ p^q)')
    plt.legend()
    plt.title('Multifractal spectrum')
    plt.show()