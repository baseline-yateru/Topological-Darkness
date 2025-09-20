import numpy as np
import CharMatrix2 as cm
import matplotlib.pyplot as plt
import Dispersion as D
import Dispersion3 as D3
import pandas as pd
import time
from functools import wraps
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import minimize_scalar


h = 4.14e-15  # Planck's constant in eV·s
c = 3e8  # Speed of light in m/s

df_pd_n = pd.read_csv('Data/Pd_n.csv')
df_pd_k = pd.read_csv('Data/Pd_k.csv')
df_pd_n['k'] = df_pd_k['k']

df_cr_n = pd.read_csv('Data/Cr_n.csv')
df_cr_k = pd.read_csv('Data/Cr_k.csv')
df_cr_n['k'] = df_cr_k['k']

df_sio2 = pd.read_csv('Data/SiO2_JAW2.csv', sep = ',', names = ['eng','e1','e2'], skiprows = 3)
df_si = pd.read_csv('Data/Si_jaw.csv', sep = ',', names = ['eng','e1','e2'], skiprows = 3)

dis_pd = D.interpolator(df_pd_n, z_col='wl', x_col='n', y_col='k')
dis_cr = D.interpolator(df_cr_n, z_col='wl', x_col='n', y_col='k')
dis_sio2 = D3.local_interpolator(df_sio2, z_col='eng', x_col='e1', y_col='e2')
dis_si = D3.local_interpolator(df_si, z_col='eng', x_col='e1', y_col='e2')


class PlotApp:
    def __init__(self, master, func):
        self.master = master
        self.master.title("Interactive Plot with Slider")

        # Create a figure
        self.fig = Figure(figsize=(6, 4), dpi=150)
        self.ax = self.fig.add_subplot(111)
        self.func = func
        # Initial plot
        
        self.ang = np.pi/3
        self.wl = np.linspace(200e-9, 1200e-9, 1000)

        mat = [self.func(self.ang, w) for w in self.wl]
        #self.line, = self.ax.plot(self.wl*1e9, np.abs(np.array([m.ref_p() for m in mat]))**2, label=r"$R_{p}$")
        #self.line2, = self.ax.plot(self.wl*1e9, np.abs(np.array([m.ref_s() for m in mat]))**2, label=r"$R_{s}$")
        self.line3, = self.ax.plot(self.wl*1e9, 180/np.pi*np.arctan(np.abs(np.array([m.elip() for m in mat]))), label=r"$\Psi$")
        self.vline = self.ax.axvline(x=0, color='red', linestyle='--', label='Optimal Wavelength')


        self.ax.legend()
        self.ax.hlines(0, 200, 1200, color='black', linestyle='--', label='Zero Line')
        self.ax.set_xlabel(r"$\lambda$ (nm)")
        self.ax.set_ylabel(r"$R_{s}$, $R_{p}$ and $\Psi$")
        self.ax.set_title(r"$R_{s}$, $R_{p}$ and $\Psi$ vs $/lambda$, $/theta$ = 0")
        self.ax.set_xlim(200,1200)
        self.ax.grid()

        # Embed the plot in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Slider to change frequency
        self.slider = ttk.Scale(master, from_=0, to=np.pi/2, orient='horizontal',
                                command=self.update_plot)
        self.slider.set(self.ang)
        self.slider.pack(fill=tk.X, padx=10, pady=10)

    def update_plot(self, val):
        self.ang = float(val)
        mat = np.array([self.func(self.ang, w) for w in self.wl])
        #self.line.set_ydata(np.abs(np.array([m.ref_p() for m in mat]))**2)
        #self.line2.set_ydata(np.abs(np.array([m.ref_s() for m in mat]))**2)
        self.line3.set_ydata(180/np.pi*np.arctan(np.abs(np.array([m.elip() for m in mat]))))


        def elip_abs(wl):
            try:
                v = np.arctan(np.abs(self.func(self.ang, wl*1e-9).elip()))
                return float(v)
            except Exception as e:
                print(f"Error at wl={wl}: {e}")
                return np.inf

        result_1 = minimize_scalar(elip_abs, bounds=(700, 1200), method='bounded')
        result_2 = minimize_scalar(elip_abs, bounds=(200, 700), method='bounded')
        lg = elip_abs(result_1.x) < elip_abs(result_2.x)

        optimal_wl = result_1.x * lg + result_2.x*(1-lg)
        print(f"Optimal wavelength for angle {self.ang:.4f} deg: {optimal_wl:.0f} nm")
        self.vline.set_xdata([optimal_wl])
        # Remove previous vertical lines and annotations

        self.ax.set_title(f"$R_{{s}}$, $R_{{p}}$ and $\\Psi$ vs $\\lambda$, $\\theta$ = {self.ang/np.pi*180:.0f}°" )
        self.canvas.draw_idle()

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' executed in {end - start:.6f} seconds")
        return result
    return wrapper

#@timeit
def extractor():
    n_list = lambda wl: np.array([1, 
                       dis_pd(1e6*wl)[0,0] + 1j*dis_pd(1e6*wl)[0,1], 
                       dis_cr(1e6*wl)[0,0] + 1j*dis_cr(1e6*wl)[0,1],
                        dis_sio2(wl)[0,0] + 1j*dis_sio2(wl)[0,1], 
                        dis_si(wl)[0,0] + 1j*dis_si(wl)[0,1]])
    d_list = np.array([3.5e-9, 1e-9, 290e-9])  # Thickness of each layer in meters
    return lambda ang, wl: cm.Substrate(ang, n_list(wl), d_list, wl)


if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root, extractor())
    root.mainloop()