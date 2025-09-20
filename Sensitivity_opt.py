import numpy as np
import CharMatrix2 as cm
import matplotlib.pyplot as plt
import DispersionC as DC
import Dispersion2 as D2
from scipy.optimize import differential_evolution
import pandas as pd

LN = 100
h = 4.14e-15  # Planck's constant in eV·s
c = 3e8       # Speed of light in m/s

# Define different p0 values to test
p0_list = [
    [2.34932649, 3.65482101, 1.46925001e-6],
    [2.34932649, 3.65482101, 1.218046875e-6],
    [2.34932649, 3.65482101,  0.6015625000000003e-6],
    [2.34932649, 3.65482101, 53.5069580078125e-6],
    [2.34932649, 3.65482101, 2.64887425e-6]

]

colors = rainbow_hex = [
    "#FF0000",  # Red
    "#FF9900",  # Orange
    "#FFFF00",  # Yellow
    "#00FF66",  # Green
    "#0066FF"   # Blue
]

ang_list = np.linspace(0,90,5)

p1 = (2.525252525252525, 4.040404040404041, 1648.87425039868e-9)  # Fixed

df_au = pd.read_csv('Data/Au_nk1.csv', sep = ',', names = ['wl','n','k'], skiprows = 3)
df_au.iloc[:, 0] *= 1e9  # Convert wavelength to meters

dis_au = D2.interpolator(df_au, z_col='wl', x_col='n', y_col='k')



def extractor(disp):
    """Returns a function that builds the multilayer system."""
    def extract(ang, wl):
        n_list = np.array([
            1.5,          # BK7
            disp(wl),     # Metal (from dispersion)
            1.33          # Water
        ])
        d_list = np.array([50e-9])  # Metal thickness
        return cm.Substrate(ang, n_list, d_list, wl)
    return extract

def extractor2(disp):
    """Returns a function that builds the multilayer system."""
    def extract(ang, wl):
        n_list = np.array([
            1.5,          # BK7
            disp(wl)[0,0] + 1j* disp(wl)[0,1],     # Metal (from dispersion)
            1.33          # Water
        ])
        d_list = np.array([50e-9])  # Metal thickness
        return cm.Substrate(ang, n_list, d_list, wl)
    return extract

def run_for_p0(p0, label=None):
    disp = DC.custom_disp(p0, p1)
    get_mat = extractor(disp)

    def elip_abs(x, w):
        mat = get_mat(x[0], w)
        return np.abs(mat.elip())

    wl_arr = np.linspace(1640, 1660, LN)
    minima = np.zeros(LN)
    for i, w in enumerate(wl_arr):
        wl_m = w * 1e-9
        result = differential_evolution(
            elip_abs,
            bounds=[(30 * np.pi / 180, 90 * np.pi / 180)],
            polish=True,
            args=(wl_m,)
        )
        minima[i] = 180 / np.pi * np.arctan(elip_abs(result.x, wl_m))

    return wl_arr, minima

def run_au(disp_au, label=None):
    disp = disp_au
    get_mat = extractor2(disp)

    def elip_abs(x, w):
        mat = get_mat(x[0], w)
        #print(mat)
        return np.abs(mat.elip())

    wl_arr = np.linspace(400, 2000, 100)
    minima = np.zeros(100)
    for i, w in enumerate(wl_arr):
        wl_m = w * 1e-9
        result = differential_evolution(
            elip_abs,
            bounds=[(30 * np.pi / 180, 90 * np.pi / 180)],
            polish=True,
            args=(wl_m,)
        )
        minima[i] = 180 / np.pi * np.arctan(elip_abs(result.x, wl_m))

    return wl_arr, minima

def main():
    plt.figure(figsize=(8, 6))

    for i, p0 in enumerate(p0_list):
        wl_arr, minima = run_for_p0(p0)
        plt.plot(wl_arr/ 1000, minima, color = colors[i], linewidth=1.5, label=f"Scissor Angle - {ang_list[i]:.1f}°")
    """
    wl_arr, minima = run_au(dis_au, label="Au Dispersion")
    plt.plot(wl_arr/ 1000, minima, color="black", linewidth=1.5, label="Au Dispersion")
    """
    plt.gca().set_xticks([1.64, 1.65, 1.66])  # Set major ticks
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.01/4))  # Set minor ticks every 0.01
    plt.xlabel(r"$\mathbf{Wavelength\  (\mu m)}$")
    plt.ylabel(r"$\mathbf{Ψ (deg)}$")
    plt.legend(bbox_to_anchor=(0.5, 0.8), loc='upper center', ncol=1)
    
    plt.grid(True, which='both')  # Show both major and minor gridlines
    plt.tight_layout()
    plt.savefig("minima_width_plot_S.tiff", format = "tiff", dpi=900)
    plt.show()




if __name__ == "__main__":
    main()
