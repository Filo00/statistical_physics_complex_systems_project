import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


# Sistema di Lorenz
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def box_counting(data2d, epsilons):
    counts = []
    for eps in epsilons:
        # Normalizza i dati in [0, 1]
        norm_data = (data2d - data2d.min(axis=0)) / (data2d.max(axis=0) - data2d.min(axis=0))
        # Griglia
        grid = np.floor(norm_data / eps).astype(int)
        # Conta box unici
        unique_boxes = set(map(tuple, grid))
        counts.append(len(unique_boxes))
    return counts

from scipy.spatial.distance import pdist, squareform

def correlation_dimension(data, radii):
    N = len(data)
    dists = squareform(pdist(data))
    np.fill_diagonal(dists, np.inf)  # evita distanze 0 con se stessi

    C_r = []
    for r in radii:
        C = np.sum(dists < r) / (N * (N - 1))
        C_r.append(C)
    return np.log(radii), np.log(C_r)



st.title("Stima della Dimensione Frattale")

# Lorenz system
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

# Simula l'attrattore
@st.cache_data
def generate_lorenz():
    t_eval = np.linspace(0, 40, 5000)
    sol = solve_ivp(lorenz, [0, 40], [1, 1, 1], t_eval=t_eval)
    return sol.y.T

data = generate_lorenz()
data2d = data[:, :2]

st.subheader("Visualizzazione dell'Attrattore")
fig = plt.figure()
plt.plot(data2d[:, 0], data2d[:, 1], lw=0.5)
plt.xlabel("x")
plt.ylabel("y")
st.pyplot(fig)

# Box-Counting
def box_counting(data2d, epsilons):
    counts = []
    for eps in epsilons:
        norm_data = (data2d - data2d.min(axis=0)) / (data2d.max(axis=0) - data2d.min(axis=0))
        grid = np.floor(norm_data / eps).astype(int)
        unique_boxes = set(map(tuple, grid))
        counts.append(len(unique_boxes))
    return counts

st.subheader("Box-Counting Dimension")

epsilons = np.logspace(-2, 0, 20)
counts = box_counting(data2d, epsilons)
log_eps = np.log(1/epsilons)
log_counts = np.log(counts)
slope_box, _ = np.polyfit(log_eps, log_counts, 1)

fig2 = plt.figure()
plt.plot(log_eps, log_counts, 'o-')
plt.title(f"Box-Counting (D ≈ {slope_box:.2f})")
plt.xlabel("log(1/ε)")
plt.ylabel("log(N(ε))")
st.pyplot(fig2)

# Grassberger–Procaccia
def correlation_dimension(data, radii):
    N = len(data)
    dists = squareform(pdist(data))
    np.fill_diagonal(dists, np.inf)
    C_r = [np.sum(dists < r) / (N * (N - 1)) for r in radii]
    return np.log(radii), np.log(C_r)

st.subheader("Correlation Dimension")

radii = np.logspace(-2, -0.5, 20)
log_r, log_C = correlation_dimension(data2d, radii)
slope_corr, _ = np.polyfit(log_r, log_C, 1)

fig3 = plt.figure()
plt.plot(log_r, log_C, 'o-')
plt.title(f"Correlation Dimension (D₂ ≈ {slope_corr:.2f})")
plt.xlabel("log(r)")
plt.ylabel("log(C(r))")
st.pyplot(fig3)


# Parametri di integrazione
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 10000)
initial_state = [1.0, 1.0, 1.0]

# Risolvi ODE
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
points = sol.y.T  # shape (10000, 3)

# Plot 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(*points.T, lw=0.5)
ax.set_title("Attrattore di Lorenz")
plt.show()


# Proiezione in 2D
data2d = points[:, :2]  # Usa x e y

# Scelte di epsilon (dimensione dei box)
epsilons = np.logspace(-2, 0, num=20, base=10.0)
counts = box_counting(data2d, epsilons)

# Plot log-log
log_eps = np.log(1/epsilons)
log_counts = np.log(counts)

# Fit lineare
slope, _ = np.polyfit(log_eps, log_counts, 1)

plt.figure(figsize=(6, 4))
plt.plot(log_eps, log_counts, 'o-', label=f"Stima D ≈ {slope:.2f}")
plt.xlabel("log(1/ε)")
plt.ylabel("log(N(ε))")
plt.title("Box-Counting Dimension Estimate (XY projection)")
plt.legend()
plt.grid(True)
plt.show()
