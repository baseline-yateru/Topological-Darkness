import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

au_n = pd.read_csv('Data/Au_n.csv')
au_k = pd.read_csv('Data/Au_k.csv')

au_n['k'] = au_k['k']

def interpolator(df, z_col='wl', x_col='n', y_col='k'):
    """
    Returns a function that interpolates (x_col, y_col) as a function of z_col using the input DataFrame.
    The returned function takes a value or array of z_col and returns interpolated (x_col, y_col).
    """
    z = df[z_col].values
    x = df[x_col].values
    y = df[y_col].values
    interp_x = interp1d(z, x, kind='quadratic', bounds_error=False, fill_value=np.nan)
    interp_y = interp1d(z, y, kind='quadratic', bounds_error=False, fill_value=np.nan)
    def interpolator(z_val):
        return np.column_stack((interp_x(z_val), interp_y(z_val)))
    return interpolator

def main():
    wl = np.linspace(0.2, 1.94, 1000)  # Wavelength range in nm


    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(au_n['n'], au_n['k'], au_n['wl'], label='Data Points')

    ref = interpolator(au_n)(wl)
    print(ref)
    print(interpolator(au_n)(0.25)[0, 0], interpolator(au_n)(0.25)[0, 1])
    ax.plot(interpolator(au_n)(wl)[:, 0], interpolator(au_n)(wl)[:, 1], wl, color='red', label='Interpolated Path')
    ax.set_xlabel('n')
    ax.set_ylabel('k')
    ax.set_zlabel('Wavelength (nm)')
    ax.set_title('3D Plot of n, k vs Wavelength')
    ax.legend()
    plt.show()

