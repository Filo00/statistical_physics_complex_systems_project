import numpy as np
import matplotlib.pyplot as plt

# --- Stima dimensione frattale: funzioni ---

def box_counting(data, box_sizes, plot=True):
    """Stima la dimensione frattale con box-counting e mostra il plot."""
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
    D = coeffs[0]
    if plot:
        plt.figure()
        plt.plot(log_inv_box_sizes, log_counts, 'o-', label='Data')
        plt.plot(log_inv_box_sizes, np.polyval(coeffs, log_inv_box_sizes), '--', label=f'Fit: slope={D:.3f}')
        plt.xlabel('log(1/box size)')
        plt.ylabel('log(N box)')
        plt.legend()
        plt.title('Box-counting plot')
        plt.show()
    return D

def correlation_dimension(data, r_vals, plot=True):
    """Stima la correlation dimension (Grassberger-Procaccia) e mostra il plot."""
    data = np.asarray(data)
    dists = np.linalg.norm(data[:, np.newaxis] - data[np.newaxis, :], axis=-1)
    N = len(data)
    C = []
    for r in r_vals:
        C.append(np.sum(dists < r) / (N * (N - 1)))
    log_r = np.log(r_vals)
    log_C = np.log(C)
    coeffs = np.polyfit(log_r, log_C, 1)
    D = coeffs[0]
    if plot:
        plt.figure()
        plt.plot(log_r, log_C, 'o-', label='Data')
        plt.plot(log_r, np.polyval(coeffs, log_r), '--', label=f'Fit: slope={D:.3f}')
        plt.xlabel('log(r)')
        plt.ylabel('log(C(r))')
        plt.legend()
        plt.title('Correlation dimension plot')
        plt.show()
    return D

def information_dimension(data, box_sizes, plot=True):
    """Stima la information dimension (D1) e mostra il plot."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1,1)
    N_dim = data.shape[1]
    S_list = []
    log_inv_box_sizes = []
    for size in box_sizes:
        bins = [np.arange(np.min(data[:, dim]), np.max(data[:, dim]) + size, size) for dim in range(N_dim)]
        hist, _ = np.histogramdd(data, bins=bins)
        p = hist.flatten() / np.sum(hist)
        p = p[p > 0]
        S = -np.sum(p * np.log(p))
        S_list.append(S)
        log_inv_box_sizes.append(np.log(1/size))
    S_list = np.array(S_list)
    log_inv_box_sizes = np.array(log_inv_box_sizes)
    coeffs = np.polyfit(log_inv_box_sizes, S_list, 1)
    D1 = coeffs[0]
    if plot:
        plt.figure()
        plt.plot(log_inv_box_sizes, S_list, 'o-', label='Data')
        plt.plot(log_inv_box_sizes, np.polyval(coeffs, log_inv_box_sizes), '--', label=f'Fit: slope={D1:.3f}')
        plt.xlabel('log(1/box size)')
        plt.ylabel('Shannon entropy S')
        plt.legend()
        plt.title('Information dimension plot')
        plt.show()
    return D1

def generalized_dimension(data, box_sizes, q=2, plot=True):
    """Stima la generalized dimension (Renyi, multifractal, Dq) e mostra il plot."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1,1)
    N_dim = data.shape[1]
    Sq_list = []
    log_box_sizes = []
    for size in box_sizes:
        bins = [np.arange(np.min(data[:, dim]), np.max(data[:, dim]) + size, size) for dim in range(N_dim)]
        hist, _ = np.histogramdd(data, bins=bins)
        p = hist.flatten() / np.sum(hist)
        p = p[p > 0]
        if q == 1:
            S = -np.sum(p * np.log(p))
        else:
            S = np.sum(p ** q)
        Sq_list.append(S)
        log_box_sizes.append(np.log(size))
    Sq_list = np.array(Sq_list)
    log_box_sizes = np.array(log_box_sizes)
    if q == 1:
        coeffs = np.polyfit(log_box_sizes, Sq_list, 1)
        Dq = coeffs[0]
    else:
        coeffs = np.polyfit(log_box_sizes, np.log(Sq_list), 1)
        Dq = coeffs[0] / (q - 1)
    if plot:
        plt.figure()
        if q == 1:
            plt.plot(log_box_sizes, Sq_list, 'o-', label='Data')
            plt.plot(log_box_sizes, np.polyval(coeffs, log_box_sizes), '--', label=f'Fit: slope={Dq:.3f}')
        else:
            plt.plot(log_box_sizes, np.log(Sq_list), 'o-', label='Data')
            plt.plot(log_box_sizes, np.polyval(coeffs, log_box_sizes), '--', label=f'Fit: slope={Dq:.3f}')
        plt.xlabel('log(box size)')
        plt.ylabel(f'log(∑p^{q})' if q != 1 else 'Shannon entropy S')
        plt.legend()
        plt.title(f'Generalized dimension D{q} plot')
        plt.show()
    return Dq

def plot_multifractal_spectrum(data, box_sizes, q_list):
    """Plot multifractal scaling: log(∑p^q) vs log(box size) for several q."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1,1)
    N_dim = data.shape[1]
    plt.figure()
    for q in q_list:
        Sq_list = []
        log_box_sizes = []
        for size in box_sizes:
            bins = [np.arange(np.min(data[:, dim]), np.max(data[:, dim]) + size, size) for dim in range(N_dim)]
            hist, _ = np.histogramdd(data, bins=bins)
            p = hist.flatten() / np.sum(hist)
            p = p[p > 0]
            if q == 1:
                S = -np.sum(p * np.log(p))
            else:
                S = np.sum(p ** q)
            Sq_list.append(S)
            log_box_sizes.append(np.log(size))
        if q == 1:
            plt.plot(log_box_sizes, Sq_list, 'o-', label=f'q={q}')
        else:
            plt.plot(log_box_sizes, np.log(Sq_list), 'o-', label=f'q={q}')
    plt.xlabel('log(box size)')
    plt.ylabel('log(∑p^q) / S(q=1)')
    plt.legend()
    plt.title('Multifractal spectrum')
    plt.show()

def nearest_neighbor_dimension(data, k=1):
    """Stima la dimensione frattale tramite nearest-neighbor (dimensione di Grassberger)."""
    data = np.asarray(data)
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(data))
    np.fill_diagonal(dists, np.inf)
    r = np.min(dists, axis=1) if k == 1 else np.partition(dists, k, axis=1)[:,k]
    mean_log_r = np.mean(np.log(r))
    estimator = -1 / mean_log_r
    return estimator

def variogram_dimension(signal, scales, plot=True):
    """Stima con il variogramma/madogramma e mostra il plot (solo segnali 1D)."""
    dims = []
    log_scales = []
    log_V = []
    signal = np.asarray(signal)
    for h in scales:
        if h <= 1:
            continue  # evita divisione per zero
        diffs = signal[h:] - signal[:-h]
        V = np.mean(np.abs(diffs))
        if V <= 0:
            continue  # evita log(0)
        dims.append(np.log(V) / np.log(h))
        log_scales.append(np.log(h))
        log_V.append(np.log(V))
    if plot and len(log_scales) > 1:
        coeffs = np.polyfit(log_scales, log_V, 1)
        plt.figure()
        plt.plot(log_scales, log_V, 'o-', label='Data')
        plt.plot(log_scales, np.polyval(coeffs, log_scales), '--', label=f'Fit: slope={coeffs[0]:.3f}')
        plt.xlabel('log(h)')
        plt.ylabel('log(V(h))')
        plt.legend()
        plt.title('Variogram plot')
        plt.show()
    return np.mean(dims) if dims else np.nan
# --- Funzioni per generare dati frattali noti ---

def sierpinski(n_iter=6):
    """Restituisce punti del triangolo di Sierpinski"""
    # Vertici del triangolo
    v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    x = np.array([0.0, 0.0])
    points = [x]
    for _ in range(3**n_iter):
        vertex = v[np.random.randint(0, 3)]
        x = (x + vertex) / 2
        points.append(x)
    return np.array(points)

def cantor_set(n_iter=10):
    """Restituisce punti del set di Cantor su [0,1]"""
    intervals = [[0.0, 1.0]]
    for _ in range(n_iter):
        new_intervals = []
        for a, b in intervals:
            third = (b - a) / 3
            new_intervals.append([a, a + third])
            new_intervals.append([b - third, b])
        intervals = new_intervals
    points = []
    for a, b in intervals:
        points.append((a + b)/2)
    return np.array(points).reshape(-1, 1)

def brownian_motion(n=1000):
    """Restituisce una traiettoria di moto browniano 1D"""
    return np.cumsum(np.random.randn(n))

def square_grid(n=32):
    """Restituisce punti su una griglia quadrata piena (dimensione 2)"""
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    return np.vstack([X.ravel(), Y.ravel()]).T

# --- MAIN ---

if __name__ == "__main__":
    np.random.seed(42)
    box_sizes = np.logspace(-2, -0.5, 10)
    r_vals = np.logspace(-2, -0.5, 10)
    scales = np.arange(2, 20)
    q_list = [0, 1, 2]

    # 1. Sierpinski triangle (dimensione teorica ~1.585)
    print("\n=== Sierpinski Triangle ===")
    data_sierpinski = sierpinski(n_iter=6)
    D_box = box_counting(data_sierpinski, box_sizes, plot=True)
    print(f"Box-counting dimension (Sierpinski): {D_box:.3f}")

    # 2. Cantor set (dimensione teorica ~0.6309)
    print("\n=== Cantor Set ===")
    data_cantor = cantor_set(n_iter=8)
    D_box = box_counting(data_cantor, box_sizes, plot=True)
    print(f"Box-counting dimension (Cantor set): {D_box:.3f}")

    # 3. Brownian motion (fronte, dimensione teorica 1.5)
    print("\n=== Brownian Motion (1D) ===")
    signal_brown = brownian_motion(n=2000)
    D_vario = variogram_dimension(signal_brown, scales, plot=True)
    print(f"Variogram dimension (Brownian): {D_vario:.3f}")

    # 4. Square grid piena (dimensione teorica 2)
    print("\n=== Square Grid (2D full) ===")
    data_grid = square_grid(n=32)
    D_box = box_counting(data_grid, box_sizes, plot=True)
    print(f"Box-counting dimension (Square grid): {D_box:.3f}")

    # 5. Dati randomici 2D (già visti)
    print("\n=== 2D Random Points ===")
    data_2d = np.random.rand(1000, 2)
    D_box = box_counting(data_2d, box_sizes, plot=True)
    print(f"Box-counting dimension (Random 2D): {D_box:.3f}")

    # 6. Dati randomici 1D (rumore bianco, dimensione ~1)
    print("\n=== 1D Random Noise ===")
    signal_1d = np.random.rand(2000)
    D_vario = variogram_dimension(signal_1d, scales, plot=True)
    print(f"Variogram dimension (Random 1D): {D_vario:.3f}")

    # Puoi aggiungere altri esempi o calcolare anche le altre dimensioni come sopra!