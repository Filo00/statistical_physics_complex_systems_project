import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import threading

# --- Funzioni frattali 2D e 1D ---

def sierpinski(n_iter=6):
    v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    x = np.array([0.0, 0.0])
    points = [x]
    for _ in range(3**n_iter):
        vertex = v[np.random.randint(0, 3)]
        x = (x + vertex) / 2
        points.append(x)
    return np.array(points)

def cantor_set(n_iter=10):
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
    return np.cumsum(np.random.randn(n))

def square_grid(n=32):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    return np.vstack([X.ravel(), Y.ravel()]).T

def sierpinski_carpet(n_iter=4):
    points = []
    def carpet(x, y, size, iter):
        if iter == 0:
            points.append([x, y])
        else:
            new_size = size / 3
            for dx in [0, 1, 2]:
                for dy in [0, 1, 2]:
                    if dx == 1 and dy == 1:
                        continue
                    carpet(x + dx*new_size, y + dy*new_size, new_size, iter-1)
    carpet(0, 0, 1, n_iter)
    return np.array(points)

def koch_curve(n_iter=4):
    def koch_rec(p1, p2, iter):
        if iter == 0:
            return [p1, p2]
        else:
            p1 = np.array(p1)
            p2 = np.array(p2)
            delta = (p2 - p1) / 3
            a = p1
            b = p1 + delta
            d = p1 + 2*delta
            angle = np.pi/3
            c = b + np.dot([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], delta)
            return (koch_rec(a, b, iter-1)[:-1] +
                    koch_rec(b, c, iter-1)[:-1] +
                    koch_rec(c, d, iter-1)[:-1] +
                    koch_rec(d, p2, iter-1))
    return np.array(koch_rec([0, 0], [1, 0], n_iter))

def dragon_curve(n_iter=12):
    points = [complex(0, 0), complex(1, 0)]
    for _ in range(n_iter):
        next_points = [points[0]]
        for i in range(len(points)-1):
            z1 = points[i]
            z2 = points[i+1]
            mid = (z1 + z2)/2 + (z2 - z1)*complex(0, 1)/2
            next_points.append(mid)
            next_points.append(z2)
        points = next_points
    arr = np.array([[p.real, p.imag] for p in points])
    return arr

# --- Frattali 3D ---

def menger_sponge(n_iter=3):
    points = [np.array([0, 0, 0])]
    size = 1
    for _ in range(n_iter):
        new_points = []
        for p in points:
            for dx in [0, 1, 2]:
                for dy in [0, 1, 2]:
                    for dz in [0, 1, 2]:
                        if sum([dx==1, dy==1, dz==1]) >= 2:
                            continue
                        new_points.append(p + size * np.array([dx, dy, dz]))
        points = new_points
        size /= 3
    points = np.array(points) * size
    return points

def sierpinski_tetrahedron(n_iter=4):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0], [0.5, np.sqrt(3)/6, np.sqrt(6)/3]])
    x = np.array([0.0, 0.0, 0.0])
    points = [x]
    for _ in range(4**n_iter):
        vertex = vertices[np.random.randint(0, 4)]
        x = (x + vertex) / 2
        points.append(x)
    return np.array(points)

def julia3d(N=40, max_iter=12, threshold=4, c=(0.355, 0.355, 0.355)):
    xs = np.linspace(-1.5, 1.5, N)
    ys = np.linspace(-1.5, 1.5, N)
    zs = np.linspace(-1.5, 1.5, N)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    points = np.vstack([X, Y, Z]).T

    Zx = X.copy()
    Zy = Y.copy()
    Zz = Z.copy()
    mask = np.ones(points.shape[0], dtype=bool)

    for i in range(max_iter):
        Zx_active = Zx[mask]
        Zy_active = Zy[mask]
        Zz_active = Zz[mask]
        xtemp = Zx_active**2 - Zy_active**2 - Zz_active**2 + c[0]
        ytemp = 2*Zx_active*Zy_active + c[1]
        ztemp = 2*Zx_active*Zz_active + c[2]

        Zx[mask] = xtemp
        Zy[mask] = ytemp
        Zz[mask] = ztemp

        diverged = (Zx[mask]**2 + Zy[mask]**2 + Zz[mask]**2) > threshold
        mask_indices = np.where(mask)[0]
        mask[mask_indices[diverged]] = False

    julia_points = points[mask]
    return julia_points

def mandelbrot3d(N=40, max_iter=12, threshold=4):
    xs = np.linspace(-2, 1, N)
    ys = np.linspace(-1.5, 1.5, N)
    zs = np.linspace(-1.5, 1.5, N)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    points = np.vstack([X, Y, Z]).T

    Cx = X.copy()
    Cy = Y.copy()
    Cz = Z.copy()
    Zx = np.zeros_like(X)
    Zy = np.zeros_like(Y)
    Zz = np.zeros_like(Z)

    mask = np.ones(points.shape[0], dtype=bool)

    for i in range(max_iter):
        Zx_active = Zx[mask]
        Zy_active = Zy[mask]
        Zz_active = Zz[mask]
        Cx_active = Cx[mask]
        Cy_active = Cy[mask]
        Cz_active = Cz[mask]
        xtemp = Zx_active**2 - Zy_active**2 - Zz_active**2 + Cx_active
        ytemp = 2*Zx_active*Zy_active + Cy_active
        ztemp = 2*Zx_active*Zz_active + Cz_active
        Zx[mask] = xtemp
        Zy[mask] = ytemp
        Zz[mask] = ztemp

        diverged = (Zx[mask]**2 + Zy[mask]**2 + Zz[mask]**2) > threshold
        mask_indices = np.where(mask)[0]
        mask[mask_indices[diverged]] = False

    mb_points = points[mask]
    return mb_points

def mandelbulb(N=40, max_iter=10, threshold=8, power=8):
    xs = np.linspace(-1.3, 1.3, N)
    ys = np.linspace(-1.3, 1.3, N)
    zs = np.linspace(-1.3, 1.3, N)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    points = np.vstack([X, Y, Z]).T

    Cx = X.copy()
    Cy = Y.copy()
    Cz = Z.copy()
    Zx = np.zeros_like(X)
    Zy = np.zeros_like(Y)
    Zz = np.zeros_like(Z)

    mask = np.ones(points.shape[0], dtype=bool)

    for i in range(max_iter):
        Zx_active = Zx[mask]
        Zy_active = Zy[mask]
        Zz_active = Zz[mask]
        Cx_active = Cx[mask]
        Cy_active = Cy[mask]
        Cz_active = Cz[mask]
        r = np.sqrt(Zx_active**2 + Zy_active**2 + Zz_active**2)
        theta = np.arctan2(np.sqrt(Zx_active**2 + Zy_active**2), Zz_active)
        phi = np.arctan2(Zy_active, Zx_active)
        rn = r**power
        thetan = theta * power
        phin = phi * power
        Zx_new = rn * np.sin(thetan) * np.cos(phin) + Cx_active
        Zy_new = rn * np.sin(thetan) * np.sin(phin) + Cy_active
        Zz_new = rn * np.cos(thetan) + Cz_active
        Zx[mask] = Zx_new
        Zy[mask] = Zy_new
        Zz[mask] = Zz_new

        diverged = (Zx[mask]**2 + Zy[mask]**2 + Zz[mask]**2) > threshold
        mask_indices = np.where(mask)[0]
        mask[mask_indices[diverged]] = False

    bulb_points = points[mask]
    return bulb_points

# --- Volumetric Mandelbulb ---
def mandelbulb_volume(N=40, power=8, max_iter=12, threshold=8):
    xs = np.linspace(-1.3, 1.3, N)
    ys = np.linspace(-1.3, 1.3, N)
    zs = np.linspace(-1.3, 1.3, N)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    values = np.zeros_like(X, dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                zx, zy, zz = X[i,j,k], Y[i,j,k], Z[i,j,k]
                cx, cy, cz = zx, zy, zz
                for it in range(max_iter):
                    r = np.sqrt(zx*zx + zy*zy + zz*zz)
                    theta = np.arctan2(np.sqrt(zx*zx + zy*zy), zz)
                    phi = np.arctan2(zy, zx)
                    rn = r**power
                    thetan = theta * power
                    phin = phi * power
                    zx_new = rn * np.sin(thetan) * np.cos(phin) + cx
                    zy_new = rn * np.sin(thetan) * np.sin(phin) + cy
                    zz_new = rn * np.cos(thetan) + cz
                    if zx_new*zx_new + zy_new*zy_new + zz_new*zz_new > threshold:
                        break
                    zx, zy, zz = zx_new, zy_new, zz_new
                values[i,j,k] = it
    return values

# --- Volumetric Julia 3D ---
def julia3d_volume(N=40, c=(0.355,0.355,0.355), max_iter=12, threshold=8):
    xs = np.linspace(-1.5,1.5,N)
    ys = np.linspace(-1.5,1.5,N)
    zs = np.linspace(-1.5,1.5,N)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    values = np.zeros_like(X, dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                zx, zy, zz = X[i,j,k], Y[i,j,k], Z[i,j,k]
                for it in range(max_iter):
                    xtemp = zx*zx - zy*zy - zz*zz + c[0]
                    ytemp = 2*zx*zy + c[1]
                    ztemp = 2*zx*zz + c[2]
                    zx, zy, zz = xtemp, ytemp, ztemp
                    if zx*zx + zy*zy + zz*zz > threshold:
                        break
                values[i,j,k] = it
    return values

# --- Volumetric Mandelbrot 3D ---
def mandelbrot3d_volume(N=40, max_iter=12, threshold=8):
    xs = np.linspace(-2.0, 1.0, N)
    ys = np.linspace(-1.5, 1.5, N)
    zs = np.linspace(-1.5, 1.5, N)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    values = np.zeros_like(X, dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                zx, zy, zz = 0.0, 0.0, 0.0
                cx, cy, cz = X[i,j,k], Y[i,j,k], Z[i,j,k]
                for it in range(max_iter):
                    xtemp = zx*zx - zy*zy - zz*zz + cx
                    ytemp = 2*zx*zy + cy
                    ztemp = 2*zx*zz + cz
                    zx, zy, zz = xtemp, ytemp, ztemp
                    if zx*zx + zy*zy + zz*zz > threshold:
                        break
                values[i,j,k] = it
    return values

# --- Metodi di stima ---
def box_counting(data, box_sizes, plot=True):
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
        plt.tight_layout()
        plt.show()
    return D

def correlation_dimension(data, r_vals, plot=True):
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
        plt.tight_layout()
        plt.show()
    return D

def variogram_dimension(signal, scales, plot=True):
    dims = []
    log_scales = []
    log_V = []
    signal = np.asarray(signal)
    for h in scales:
        if h <= 1:
            continue
        diffs = signal[h:] - signal[:-h]
        V = np.mean(np.abs(diffs))
        if V <= 0:
            continue
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
        plt.tight_layout()
        plt.show()
    return np.mean(dims) if dims else np.nan

# --- Pagina Rendering Volumetrico Plotly ---
class VolumetricFractalWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Rendering Volumetrico Frattale 3D")
        self.geometry("430x420")
        self.resizable(False, False)

        self.fractal_types = ["Mandelbulb", "Julia 3D", "Mandelbrot 3D"]
        self.fractal_var = tk.StringVar(value=self.fractal_types[0])
        self.N_var = tk.IntVar(value=48)
        self.power_var = tk.IntVar(value=8)
        self.max_iter_var = tk.IntVar(value=12)
        self.threshold_var = tk.DoubleVar(value=8.0)
        self.julia_cx = tk.DoubleVar(value=0.355)
        self.julia_cy = tk.DoubleVar(value=0.355)
        self.julia_cz = tk.DoubleVar(value=0.355)

        row = 0
        tk.Label(self, text="Tipo di frattale volumetrico:").grid(row=row, column=0, sticky="e")
        ttk.Combobox(self, textvariable=self.fractal_var, values=self.fractal_types, state='readonly', width=18).grid(row=row, column=1, sticky="w")

        row += 1
        tk.Label(self, text="Risoluzione N:").grid(row=row, column=0, sticky="e")
        tk.Entry(self, textvariable=self.N_var, width=8).grid(row=row, column=1, sticky="w")

        row += 1
        tk.Label(self, text="Iterazioni massime:").grid(row=row, column=0, sticky="e")
        tk.Entry(self, textvariable=self.max_iter_var, width=8).grid(row=row, column=1, sticky="w")

        row += 1
        tk.Label(self, text="Soglia (threshold):").grid(row=row, column=0, sticky="e")
        tk.Entry(self, textvariable=self.threshold_var, width=8).grid(row=row, column=1, sticky="w")

        row += 1
        self.power_row = row
        tk.Label(self, text="Potenza (solo Mandelbulb):").grid(row=row, column=0, sticky="e")
        self.power_entry = tk.Entry(self, textvariable=self.power_var, width=8)
        self.power_entry.grid(row=row, column=1, sticky="w")

        row += 1
        self.julia_row = row
        tk.Label(self, text="c Julia (x, y, z):").grid(row=row, column=0, sticky="e")
        frame_c = tk.Frame(self)
        frame_c.grid(row=row, column=1, sticky="w")
        tk.Entry(frame_c, textvariable=self.julia_cx, width=5).pack(side=tk.LEFT)
        tk.Entry(frame_c, textvariable=self.julia_cy, width=5).pack(side=tk.LEFT)
        tk.Entry(frame_c, textvariable=self.julia_cz, width=5).pack(side=tk.LEFT)

        row += 1
        tk.Label(self, text="Consiglio: N=40-56, iter=12, threshold=8").grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        tk.Button(self, text="Mostra rendering volumetrico", command=self.start_rendering).grid(row=row, column=0, columnspan=2, pady=14)

        self.progress = tk.Label(self, text="")
        self.progress.grid(row=row+1, column=0, columnspan=2)

        self.fractal_var.trace_add('write', self.update_fields)
        self.update_fields()

    def update_fields(self, *args):
        if self.fractal_var.get() == "Mandelbulb":
            self.power_entry.config(state="normal")
            for child in self.grid_slaves(row=self.julia_row):
                child.config(state="disabled")
        elif self.fractal_var.get() == "Julia 3D":
            self.power_entry.config(state="disabled")
            for child in self.grid_slaves(row=self.julia_row):
                child.config(state="normal")
        else:
            self.power_entry.config(state="disabled")
            for child in self.grid_slaves(row=self.julia_row):
                child.config(state="disabled")

    def start_rendering(self):
        t = threading.Thread(target=self.render)
        t.start()

    def render(self):
        self.progress.config(text="Calcolo in corso... attendere (può richiedere molto tempo)")
        self.update()
        ftype = self.fractal_var.get()
        N = self.N_var.get()
        max_iter = self.max_iter_var.get()
        threshold = self.threshold_var.get()
        if ftype == "Mandelbulb":
            power = self.power_var.get()
            volume = mandelbulb_volume(N=N, power=power, max_iter=max_iter, threshold=threshold)
            xs = np.linspace(-1.3, 1.3, N)
            ys = np.linspace(-1.3, 1.3, N)
            zs = np.linspace(-1.3, 1.3, N)
        elif ftype == "Julia 3D":
            c = (self.julia_cx.get(), self.julia_cy.get(), self.julia_cz.get())
            volume = julia3d_volume(N=N, c=c, max_iter=max_iter, threshold=threshold)
            xs = np.linspace(-1.5, 1.5, N)
            ys = np.linspace(-1.5, 1.5, N)
            zs = np.linspace(-1.5, 1.5, N)
        elif ftype == "Mandelbrot 3D":
            volume = mandelbrot3d_volume(N=N, max_iter=max_iter, threshold=threshold)
            xs = np.linspace(-2.0, 1.0, N)
            ys = np.linspace(-1.5, 1.5, N)
            zs = np.linspace(-1.5, 1.5, N)
        else:
            messagebox.showerror("Errore", "Tipo di frattale non supportato.")
            return
        fig = go.Figure(data=go.Volume(
            x=xs.repeat(N*N),
            y=np.tile(ys.repeat(N), N),
            z=np.tile(zs, N*N),
            value=volume.flatten(),
            isomin=int(max_iter*0.35), isomax=int(max_iter*0.95),
            opacity=0.08,
            surface_count=22,
            colorscale='magma',
        ))
        fig.update_layout(title=f"{ftype} volumetrico (N={N})")
        self.progress.config(text="Apro visualizzazione Plotly...")
        self.update()
        fig.show()
        self.progress.config(text="Rendering terminato")

# --- GUI PRINCIPALE ---
class FractalGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fractal Dimension Estimator + Rendering Volumetrico")
        self.geometry("720x800")
        self.resizable(False, False)

        self.fractals_2d = [
            "Random 2D", "Sierpinski", "Cantor", "Brownian", "Square Grid",
            "Sierpinski Carpet", "Koch Curve", "Dragon Curve"
        ]
        self.fractals_3d = [
            "Menger Sponge", "Sierpinski Tetrahedron", "Julia Set 3D", "Mandelbrot Set 3D", "Mandelbulb"
        ]
        self.methods = [
            ("Box-counting", box_counting, "2d"),
            ("Correlation dimension", correlation_dimension, "2d"),
            ("Variogram (1D)", variogram_dimension, "1d"),
        ]

        # --- Frame 2D ---
        frame2d = tk.LabelFrame(self, text="Frattali 1D / 2D", padx=6, pady=6)
        frame2d.place(x=10, y=10, width=320, height=370)

        tk.Label(frame2d, text="Fractal type:").grid(row=0, column=0, sticky="e")
        self.fractal_var = tk.StringVar(value=self.fractals_2d[0])
        ttk.Combobox(frame2d, textvariable=self.fractal_var, values=self.fractals_2d, state='readonly', width=20).grid(row=0, column=1, sticky="w")

        tk.Label(frame2d, text="Estimation method:").grid(row=1, column=0, sticky="e")
        self.method_var = tk.StringVar(value=self.methods[0][0])
        ttk.Combobox(frame2d, textvariable=self.method_var, values=[m[0] for m in self.methods], state='readonly', width=20).grid(row=1, column=1, sticky="w")

        tk.Label(frame2d, text="N points / iter:").grid(row=2, column=0, sticky="e")
        self.n_var = tk.IntVar(value=1000)
        tk.Entry(frame2d, textvariable=self.n_var, width=10).grid(row=2, column=1, sticky="w")

        tk.Label(frame2d, text="Min scale (es: 0.01):").grid(row=3, column=0, sticky="e")
        self.min_scale_var = tk.DoubleVar(value=0.01)
        tk.Entry(frame2d, textvariable=self.min_scale_var, width=10).grid(row=3, column=1, sticky="w")

        tk.Label(frame2d, text="Max scale (es: 0.3):").grid(row=4, column=0, sticky="e")
        self.max_scale_var = tk.DoubleVar(value=0.3)
        tk.Entry(frame2d, textvariable=self.max_scale_var, width=10).grid(row=4, column=1, sticky="w")

        tk.Label(frame2d, text="N scales:").grid(row=5, column=0, sticky="e")
        self.n_scales_var = tk.IntVar(value=10)
        tk.Entry(frame2d, textvariable=self.n_scales_var, width=10).grid(row=5, column=1, sticky="w")

        tk.Button(frame2d, text="Generate & Estimate", command=self.run_2d).grid(row=6, column=0, columnspan=2, pady=8)
        tk.Button(frame2d, text="Mostra Frattale", command=self.show_fractal_2d).grid(row=7, column=0, columnspan=2, pady=4)
        tk.Button(frame2d, text="Salva Frattale", command=self.save_fractal_image_2d).grid(row=8, column=0, columnspan=2, pady=2)
        tk.Button(frame2d, text="Salva Curva Dimensionale", command=self.save_dimension_curve_2d).grid(row=9, column=0, columnspan=2, pady=2)

        self.result_label_2d = tk.Label(frame2d, text="Dimension: --", font=("Arial", 11, "bold"))
        self.result_label_2d.grid(row=10, column=0, columnspan=2, pady=8)

        # --- Frame 3D ---
        frame3d = tk.LabelFrame(self, text="Frattali 3D", padx=6, pady=6)
        frame3d.place(x=340, y=10, width=340, height=540)

        tk.Label(frame3d, text="Fractal 3D type:").grid(row=0, column=0, sticky="e")
        self.fractal3d_var = tk.StringVar(value=self.fractals_3d[0])
        ttk.Combobox(frame3d, textvariable=self.fractal3d_var, values=self.fractals_3d, state='readonly', width=20).grid(row=0, column=1, sticky="w")

        tk.Label(frame3d, text="Iterazioni/Resolution:").grid(row=1, column=0, sticky="e")
        self.n3d_var = tk.IntVar(value=3)  # default for Menger/Sierpinski tetrahedron
        self.n3d_entry = tk.Entry(frame3d, textvariable=self.n3d_var, width=10)
        self.n3d_entry.grid(row=1, column=1, sticky="w")

        tk.Label(frame3d, text="N grid (Julia/Mandelbrot/Bulb)").grid(row=2, column=0, sticky="e")
        self.juliaN_var = tk.IntVar(value=48)
        tk.Entry(frame3d, textvariable=self.juliaN_var, width=10).grid(row=2, column=1, sticky="w")

        tk.Label(frame3d, text="c Julia (x,y,z):").grid(row=3, column=0, sticky="e")
        self.julia_cx = tk.DoubleVar(value=0.355)
        self.julia_cy = tk.DoubleVar(value=0.355)
        self.julia_cz = tk.DoubleVar(value=0.355)
        frame_c = tk.Frame(frame3d)
        frame_c.grid(row=3, column=1, sticky="w")
        tk.Entry(frame_c, textvariable=self.julia_cx, width=5).pack(side=tk.LEFT)
        tk.Entry(frame_c, textvariable=self.julia_cy, width=5).pack(side=tk.LEFT)
        tk.Entry(frame_c, textvariable=self.julia_cz, width=5).pack(side=tk.LEFT)

        tk.Label(frame3d, text="max_iter (tutti 3D):").grid(row=4, column=0, sticky="e")
        self.julia_iter = tk.IntVar(value=12)
        tk.Entry(frame3d, textvariable=self.julia_iter, width=10).grid(row=4, column=1, sticky="w")

        tk.Label(frame3d, text="threshold (tutti 3D):").grid(row=5, column=0, sticky="e")
        self.julia_thr = tk.DoubleVar(value=8.0)
        tk.Entry(frame3d, textvariable=self.julia_thr, width=10).grid(row=5, column=1, sticky="w")

        tk.Label(frame3d, text="Potenza Mandelbulb:").grid(row=6, column=0, sticky="e")
        self.bulb_power = tk.IntVar(value=8)
        tk.Entry(frame3d, textvariable=self.bulb_power, width=10).grid(row=6, column=1, sticky="w")

        tk.Button(frame3d, text="Mostra Frattale 3D", command=self.show_fractal_3d).grid(row=7, column=0, columnspan=2, pady=8)
        tk.Button(frame3d, text="Salva Frattale 3D", command=self.save_fractal_image_3d).grid(row=8, column=0, columnspan=2, pady=2)

        self.result_label_3d = tk.Label(frame3d, text="Dimension: --", font=("Arial", 11, "bold"))
        self.result_label_3d.grid(row=9, column=0, columnspan=2, pady=8)

        # Suggerimenti
        tips = (
            "Suggerimenti frattali 3D:\n"
            "• Mandelbulb: N=40-56, power=8, threshold=8, max_iter=12.\n"
            "• Julia 3D: N=40-56, c=(0.355,0.355,0.355), threshold=4-8, max_iter=10-16.\n"
            "• Mandelbrot 3D: N=40-56, threshold=8, max_iter=12.\n"
            "• Occhio: N alto = più dettagli (ma più lento!)"
        )
        tk.Label(frame3d, text=tips, justify="left", font=("Arial", 9), fg="brown").grid(row=10, column=0, columnspan=2, pady=12)

        # --- Bottone Rendering Volumetrico ---
        tk.Button(self, text="Rendering Volumetrico 3D (Plotly)", font=("Arial", 12, "bold"),
                  command=self.open_volumetric_window, bg="#d5e8d4").place(x=230, y=570, width=260, height=40)

        tk.Label(self, text="Copilot | 2024", font=("Arial", 9, "italic")).place(x=570, y=770)

        # --- Stato ---
        self.last_data_2d = None
        self.last_type_2d = None
        self.fractal_fig_2d = None
        self.dimension_fig_2d = None
        self.last_data_3d = None
        self.fractal_fig_3d = None

    def open_volumetric_window(self):
        VolumetricFractalWindow(self)

    # --- 2D/1D ---
    def generate_fractal_2d(self):
        fractal = self.fractal_var.get()
        n = self.n_var.get()
        if fractal == "Random 2D":
            data = np.random.rand(n, 2)
            data_type = "2d"
        elif fractal == "Sierpinski":
            data = sierpinski(int(np.log(n)/np.log(3)))
            data_type = "2d"
        elif fractal == "Cantor":
            data = cantor_set(int(np.log(n)/np.log(2)))
            data_type = "1d"
        elif fractal == "Brownian":
            data = brownian_motion(n)
            data_type = "1d"
        elif fractal == "Square Grid":
            grid_size = int(np.sqrt(n))
            data = square_grid(grid_size)
            data_type = "2d"
        elif fractal == "Sierpinski Carpet":
            n_iter = min(5, int(np.log(n)/np.log(8)))
            data = sierpinski_carpet(n_iter)
            data_type = "2d"
        elif fractal == "Koch Curve":
            n_iter = min(7, int(np.log(n)/np.log(4)))
            data = koch_curve(n_iter)
            data_type = "2d"
        elif fractal == "Dragon Curve":
            n_iter = min(15, int(np.log(n)/np.log(2)))
            data = dragon_curve(n_iter)
            data_type = "2d"
        else:
            messagebox.showerror("Error", "Unknown fractal type.")
            return None, None
        self.last_data_2d = data
        self.last_type_2d = data_type
        return data, data_type

    def show_fractal_2d(self):
        data, data_type = self.generate_fractal_2d()
        if data is None:
            return
        self.fractal_fig_2d = plt.figure()
        if data_type == "2d":
            plt.scatter(data[:,0], data[:,1], s=1, alpha=0.7)
            plt.xlabel("x")
            plt.ylabel("y")
        else:
            plt.plot(data, '.', markersize=2)
            plt.xlabel("Index")
            plt.ylabel("Value")
        plt.title(f"Frattale: {self.fractal_var.get()}")
        plt.tight_layout()
        plt.show()

    def run_2d(self):
        data, data_type = self.generate_fractal_2d()
        if data is None:
            return
        method = self.method_var.get()
        min_scale = self.min_scale_var.get()
        max_scale = self.max_scale_var.get()
        n_scales = self.n_scales_var.get()

        if method == "Variogram (1D)":
            scales = np.arange(max(2, int(min_scale * len(data))), int(max_scale * len(data)), max(1, int((max_scale - min_scale) * len(data) / n_scales)))
            if len(scales) < 2:
                scales = np.arange(2, 20)
        else:
            scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)

        for m_name, m_func, m_type in self.methods:
            if m_name == method:
                if m_type == "2d" and data_type == "1d":
                    messagebox.showerror("Error", f"{method} requires 2D data.")
                    return
                if m_type == "1d" and data_type == "2d":
                    messagebox.showerror("Error", f"{method} requires 1D data.")
                    return
                try:
                    plt.close('all')
                    self.dimension_fig_2d = plt.figure()
                    D = m_func(data, scales, plot=True)
                except Exception as e:
                    messagebox.showerror("Error", f"Computation failed: {e}")
                    return
                self.result_label_2d.config(text=f"Dimension: {D:.3f}" if D==D else "Dimension: --")
                return

    def save_fractal_image_2d(self):
        data, data_type = self.generate_fractal_2d()
        if data is None:
            return
        fig = plt.figure()
        if data_type == "2d":
            plt.scatter(data[:,0], data[:,1], s=1, alpha=0.7)
            plt.xlabel("x")
            plt.ylabel("y")
        else:
            plt.plot(data, '.', markersize=2)
            plt.xlabel("Index")
            plt.ylabel("Value")
        plt.title(f"Frattale: {self.fractal_var.get()}")
        plt.tight_layout()
        file_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("PNG files","*.png"),("All Files","*.*")])
        if file_path:
            fig.savefig(file_path, dpi=300)
            messagebox.showinfo("Salvato", f"Immagine frattale salvata:\n{file_path}")
        plt.close(fig)

    def save_dimension_curve_2d(self):
        if self.dimension_fig_2d is None:
            messagebox.showwarning("Attenzione", "Devi prima stimare la dimensione frattale.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("PNG files","*.png"),("All Files","*.*")])
        if file_path:
            self.dimension_fig_2d.savefig(file_path, dpi=300)
            messagebox.showinfo("Salvato", f"Curva dimensionale salvata:\n{file_path}")

    # --- 3D ---
    def generate_fractal_3d(self):
        fractal = self.fractal3d_var.get()
        if fractal == "Menger Sponge":
            n_iter = self.n3d_var.get()
            data = menger_sponge(n_iter)
        elif fractal == "Sierpinski Tetrahedron":
            n_iter = self.n3d_var.get()
            data = sierpinski_tetrahedron(n_iter)
        elif fractal == "Julia Set 3D":
            N = self.juliaN_var.get()
            max_iter = self.julia_iter.get()
            thr = self.julia_thr.get()
            c = (self.julia_cx.get(), self.julia_cy.get(), self.julia_cz.get())
            data = julia3d(N=N, max_iter=max_iter, threshold=thr, c=c)
        elif fractal == "Mandelbrot Set 3D":
            N = self.juliaN_var.get()
            max_iter = self.julia_iter.get()
            thr = self.julia_thr.get()
            data = mandelbrot3d(N=N, max_iter=max_iter, threshold=thr)
        elif fractal == "Mandelbulb":
            N = self.juliaN_var.get()
            max_iter = self.julia_iter.get()
            thr = self.julia_thr.get()
            power = self.bulb_power.get()
            data = mandelbulb(N=N, max_iter=max_iter, threshold=thr, power=power)
        else:
            messagebox.showerror("Error", "Unknown 3D fractal type.")
            return None
        self.last_data_3d = data
        return data

    def show_fractal_3d(self):
        data = self.generate_fractal_3d()
        if data is None:
            return
        self.fractal_fig_3d = plt.figure(figsize=(8,8))
        ax = self.fractal_fig_3d.add_subplot(111, projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2], s=1, alpha=0.5)
        ax.set_title(f"Frattale 3D: {self.fractal3d_var.get()}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.tight_layout()
        plt.show()

    def save_fractal_image_3d(self):
        data = self.generate_fractal_3d()
        if data is None:
            return
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2], s=1, alpha=0.5)
        ax.set_title(f"Frattale 3D: {self.fractal3d_var.get()}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.tight_layout()
        file_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("PNG files","*.png"),("All Files","*.*")])
        if file_path:
            fig.savefig(file_path, dpi=300)
            messagebox.showinfo("Salvato", f"Immagine frattale 3D salvata:\n{file_path}")
        plt.close(fig)

if __name__ == "__main__":
    app = FractalGUI()
    app.mainloop()