import numpy as np
import CharMatrix as cm
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





def main():
    ang = np.linspace(40/180*np.pi, 80/180*np.pi, 100)
    wl = np.linspace(250e-9, 1700e-9, 100)

    ang_mesh, wl_mesh = np.meshgrid(ang, wl)

    n_list = lambda wl: np.array([1, 
                       dis_pd(1e6*wl)[0,0] + 1j*dis_pd(1e6*wl)[0,1], 
                       dis_cr(1e6*wl)[0,0] + 1j*dis_cr(1e6*wl)[0,1],
                        dis_sio2(wl)[0,0] + 1j*dis_sio2(wl)[0,1], 
                        dis_si(wl)[0,0] + 1j*dis_si(wl)[0,1]])
    d_list = np.array([3.5e-9, 1e-9, 290e-9])  # Thickness of each layer in meters


    M =  np.array([[cm.Substrate(a, n_list(w), d_list, w) for w in wl] for a in ang])

    #rs_arr = np.array([[m.ref_s() for m in r] for r in M])
    #rp_arr = np.array([[m.ref_p() for m in r] for r in M ])
    el_arr = np.array([[m.elip() for m in r] for r in M])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    #print([m.angls for m in M])
    #ax.plot(wl, np.abs(rs_arr)**2, label='Reflectance s-polarized')
    #surf = ax.plot_surface(ang_mesh, 1e9*wl_mesh, np.abs(rp_arr.T)**2, label='Reflectance p-polarized', cmap='viridis', alpha=0.7)
    base = ax.plot_surface(1e9*wl_mesh, ang_mesh[:,::-1], np.zeros(np.shape(ang_mesh)), color='black', alpha=0.3)
    surf = ax.plot_surface(1e9*wl_mesh, ang_mesh, np.arctan(np.abs(el_arr).T), cmap='viridis',
                            alpha=0.7, label=r'Ellipsometric Ratio $\rho = r_p / r_s$', edgecolor = 'k', rstride=3, cstride=3)
    #plt.plot(ang, np.angle(el_arr), label=r'Ellipsometric Ratio $\rho = r_p / r_s$')

    #plt.hlines(0, wl[0], wl[-1], color='black', linestyle='--', label='Zero Line')
    fig.colorbar(surf)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Incident Angle (radians)')
    ax.set_zlabel(r'$\Psi$ (Rad)')
    #plt.title(r'$\Psi$, $R_{s}$ & $R_{p}$ vs Wavelength')
    plt.legend()
    plt.grid()
    #plt.savefig("Figure_1.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    main()

