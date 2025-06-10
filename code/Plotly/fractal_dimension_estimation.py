import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

def box_counting(data, box_sizes):
    """Stima la dimensione frattale tramite il metodo box-counting."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1,1)
    N_dim = data.shape[1]
    counts = []
    for size in box_sizes:
        bins = [np.arange(np.min(data[:, dim]), np.max(data[:, dim]) + size, size) for dim in range(N_dim)]
        hist, _ = np.histogramdd(data, bins=bins)
        counts.append(np.sum(hist > 0))
    coeffs = np.polyfit(np.log(1/np.array(box_sizes)), np.log(counts), 1)
    return coeffs[0]

def correlation_dimension(data, r_vals):
    """Stima la correlation dimension (Grassberger-Procaccia)."""
    data = np.asarray(data)
    dists = pdist(data)
    N = len(data)
    C = []
    for r in r_vals:
        C.append(np.sum(dists < r) * 2.0 / (N * (N - 1)))
    coeffs = np.polyfit(np.log(r_vals), np.log(C), 1)
    return coeffs[0]

def information_dimension(data, box_sizes):
    """Stima la information dimension (D1) usando l'entropia di Shannon."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1,1)
    N_dim = data.shape[1]
    D1_list = []
    for size in box_sizes:
        bins = [np.arange(np.min(data[:, dim]), np.max(data[:, dim]) + size, size) for dim in range(N_dim)]
        hist, _ = np.histogramdd(data, bins=bins)
        p = hist.flatten() / np.sum(hist)
        p = p[p > 0]
        S = -np.sum(p * np.log(p))
        D1_list.append(S / np.log(1/size))
    return np.mean(D1_list)

def generalized_dimension(data, box_sizes, q):
    """Stima la generalized dimension (Renyi, multifractal, Dq)."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1,1)
    N_dim = data.shape[1]
    Dq_list = []
    for size in box_sizes:
        bins = [np.arange(np.min(data[:, dim]), np.max(data[:, dim]) + size, size) for dim in range(N_dim)]
        hist, _ = np.histogramdd(data, bins=bins)
        p = hist.flatten() / np.sum(hist)
        p = p[p > 0]
        if q == 1:
            # Information dimension
            S = -np.sum(p * np.log(p))
            Dq = S / np.log(1/size)
        else:
            S = np.sum(p ** q)
            Dq = 1 / (q - 1) * np.log(S) / np.log(size)
        Dq_list.append(Dq)
    return np.mean(Dq_list)

def nearest_neighbor_dimension(data, k=1):
    """Stima la dimensione frattale tramite nearest-neighbor (dimensione di Grassberger)."""
    data = np.asarray(data)
    dists = squareform(pdist(data))
    np.fill_diagonal(dists, np.inf)
    r = np.min(dists, axis=1) if k == 1 else np.partition(dists, k, axis=1)[:,k]
    N = len(data)
    mean_log_r = np.mean(np.log(r))
    estimator = -1 / mean_log_r
    return estimator

def variogram_dimension(signal, scales):
    """Stima con il variogramma/madogramma, utile per segnali 1D."""
    dims = []
    signal = np.asarray(signal)
    for h in scales:
        if h <= 1:
            continue  # evita divisione per zero
        diffs = signal[h:] - signal[:-h]
        V = np.mean(np.abs(diffs))
        if V <= 0:
            continue  # evita log(0)
        dims.append(np.log(V) / np.log(h))
    return np.mean(dims) if dims else np.nan

# Esempio di utilizzo
if __name__ == "__main__":
    # Esempio: set di punti su una linea frattale (es: Cantor set, linea, Sierpinski, ...).
    N = 1000
    data = np.random.rand(N, 2)  # Sostituisci con i tuoi dati

    # Box-counting
    box_sizes = np.logspace(-2, -0.5, 10)
    D_box = box_counting(data, box_sizes)
    print("Box-counting dimension:", D_box)

    # Correlation dimension
    r_vals = np.logspace(-2, -0.5, 10)
    D_corr = correlation_dimension(data, r_vals)
    print("Correlation dimension:", D_corr)

    # Information dimension
    D_info = information_dimension(data, box_sizes)
    print("Information dimension:", D_info)

    # Generalized dimension (D2)
    D2 = generalized_dimension(data, box_sizes, q=2)
    print("Generalized dimension D2:", D2)

    # Nearest-neighbor
    D_nn = nearest_neighbor_dimension(data)
    print("Nearest-neighbor dimension:", D_nn)

    # Variogram (solo per segnali 1D)
    signal = np.random.rand(N)
    scales = np.arange(1, 20)
    D_vario = variogram_dimension(signal, scales)
    print("Variogram dimension (1D):", D_vario)