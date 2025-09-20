import numpy as np
import CharMatrix2 as cm
import matplotlib.pyplot as plt
import DispersionC as DC
import time
from functools import wraps
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution


h = 4.14e-15  # Planck's constant in eV·s
c = 3e8  # Speed of light in m/s

p0 = (1.515, 3.0303, 1500*1e-9)
p1 = (2.525, 4.0404, 1649*1e-9)

disp = DC.custom_disp(p0, p1)

#dis_au = D.interpolator(au_n , z_col='wl', x_col='n', y_col='k')


class PlotApp:
    def __init__(self, master, func):
        self.master = master
        self.master.title("Interactive Plot with Slider")

        # Create a figure
        self.fig = Figure(figsize=(6, 4), dpi=150)
        self.ax = self.fig.add_subplot(111)
        self.func = func
        # Initial plot

        self.ang = np.linspace(0, np.pi/2, 1000)
        self.wl = 1200e-9

        mat = [self.func(a, self.wl) for a in self.ang]

        #self.line, = self.ax.plot(self.wl*1e9, np.abs(np.array([m.ref_p() for m in mat]))**2, label=r"$R_{p}$")
        #self.line2, = self.ax.plot(self.wl*1e9, np.abs(np.array([m.ref_s() for m in mat]))**2, label=r"$R_{s}$")
        self.line3, = self.ax.plot(self.ang, 180/np.pi*np.arctan(np.abs(np.array([m.elip() for m in mat]))), label=r"$\Psi$")
        self.vline = self.ax.axvline(x=0, color='red', linestyle='--', label='Optimal Wavelength')


        self.ax.legend()
        self.ax.hlines(0, 0, np.pi/2, color='black', linestyle='--', label='Zero Line')
        self.ax.set_xlabel(r"$\lambda$ (nm)")
        self.ax.set_ylabel(r"$R_{s}$, $R_{p}$ and $\Psi$")
        self.ax.set_title(r"$R_{s}$, $R_{p}$ and $\Psi$ vs $/lambda$, $/theta$ = 0")
        self.ax.set_xlim(0, np.pi/2)
        self.ax.grid()

        # Embed the plot in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Slider to change frequency
        self.slider = ttk.Scale(master, from_=200e-9, to=3000e-9, orient='horizontal',
                                command=self.update_plot)
        self.slider.set(self.wl)
        self.slider.pack(fill=tk.X, padx=10, pady=10)

    def update_plot(self, val):
        self.wl = float(val)
        mat = np.array([self.func(a, self.wl) for a in self.ang])
        #self.line.set_ydata(np.abs(np.array([m.ref_p() for m in mat]))**2)
        #self.line2.set_ydata(np.abs(np.array([m.ref_s() for m in mat]))**2)
        self.line3.set_ydata(180/np.pi*np.arctan(np.abs(np.array([m.elip() for m in mat]))))


        def elip_abs(ang):
            mat = self.func(ang, self.wl)
            return np.abs(mat.elip())



        result_1 = differential_evolution(lambda x: elip_abs(x[0]), bounds=[(50*np.pi/180, 90*np.pi/180)], polish=True)

        optimal_ang = result_1.x[0]
        print(f"Optimal wavelength for wavelength {self.wl * 1e9:.4f} nm: {180/np.pi*optimal_ang:.0f} deg")
        print(f"Value at minimum: {elip_abs(optimal_ang):.4f}")
        self.vline.set_xdata([optimal_ang])
        # Remove previous vertical lines and annotations

        self.ax.set_title(f"$R_{{s}}$, $R_{{p}}$ and $\\Psi$ vs $\\lambda$, $\\lambda$ = {self.wl*1e9:.4f}" )
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
    #Set Layer Structure in Refractive Index

    n_list = lambda wl: np.array([
                        1.5, #BK7
                        disp(wl), #Au
                        1.33]) #Water
    
    d_list = np.array([50e-9])  # Thickness of each layer in meters
    return lambda ang, wl: cm.Substrate(ang, n_list(wl), d_list, wl)


if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root, extractor())
    root.mainloop()