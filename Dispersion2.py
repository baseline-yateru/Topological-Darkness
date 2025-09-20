import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

df = pd.read_csv('Data/Au_nk1.csv', sep = ',', names = ['wl','n','k'], skiprows = 3)

def interpolator(df, z_col='wl', x_col='n', y_col='k'):
    """
    Returns a function that interpolates (x_col, y_col) as a function of z_col using the input DataFrame.
    The returned function takes a value or array of z_col and returns interpolated (x_col, y_col).
    """
    z = df[z_col].to_numpy(dtype=np.float64)
    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)

    interp_x = interp1d(z, x, kind='cubic', bounds_error=False, fill_value=np.nan, assume_sorted=False)
    interp_y = interp1d(z, y, kind='cubic', bounds_error=False, fill_value=np.nan, assume_sorted=False)

    def interpolator(z_val):
        z_val = np.asarray(z_val, dtype=np.float64)
        return np.column_stack((interp_x(z_val), interp_y(z_val)))

    return interpolator


"""
def interpolator(df, z_col='wl', x_col='n', y_col='k'):
    
"""
    #Returns a function that interpolates (x_col, y_col) as a function of z_col using the input DataFrame.
    #The returned function takes a value or array of z_col and returns interpolated (x_col, y_col).
"""
    
    z = df[z_col].values
    x = df[x_col].values
    y = df[y_col].values
    interp_x = interp1d(z, x, kind='linear', bounds_error=False, fill_value=np.nan)
    interp_y = interp1d(z, y, kind='linear', bounds_error=False, fill_value=np.nan)
    def interpolator(z_val):
        return np.column_stack((interp_x(z_val), interp_y(z_val)))
    return interpolator
"""

def plotter():
    """
    Generates a 3D scatter plot and interpolated path of refractive index (n), extinction coefficient (k), 
    and wavelength (wl) using data from a DataFrame `df` and an interpolation function `interpolator`.
    The function:
    - Creates a wavelength range from 250 nm to 1000 nm.
    - Plots the original data points (n, k, wl) in 3D.
    - Plots the interpolated path of (n, k) over the wavelength range.
    - Labels axes and adds a legend and title.
    - Displays the plot.
    Assumes `df` is a DataFrame with columns 'n', 'k', and 'wl', and `interpolator` is a callable that 
    returns interpolated (n, k) values for given wavelengths.
    """

    wl = np.linspace(250, 1000, 1000)  # Wavelength range in nm
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['n'], df['k'], df['wl'], label='Data Points')
    dis = interpolator(df)(wl)
    ax.plot(dis[:, 0], dis[:, 1], wl, color='red', label='Interpolated Path')
    ax.set_xlabel('n')
    ax.set_ylabel('k')
    ax.set_zlabel('Wavelength (nm)')
    ax.set_title('3D Plot of n, k vs Wavelength')
    ax.legend()
    plt.show()

def plot_nk_vs_wl():
    """
    Plots the refractive index (n) and extinction coefficient (k) as functions of wavelength.
    This function interpolates the n and k values over a specified wavelength range and
    visualizes both the original data points and the interpolated curves.
    The plot includes:
        - Original n and k data points from the dataframe `df`
        - Interpolated n and k curves over the wavelength range 100 nm to 1700 nm
    Assumes:
        - `df` is a pandas DataFrame containing columns 'wl', 'n', and 'k'
        - `interpolator` is a function that returns an interpolation callable for the dataframe
        - `np` (NumPy) and `plt` (Matplotlib) are imported
    Returns:
        None. Displays the plot.
    """

    wl = np.linspace(100, 1700, 1000)
    interp = interpolator(df)
    interpolated = interp(wl)
    plt.figure(figsize=(8, 6))
    plt.plot(df['wl'], df['n'], 'o', label='n Data')
    plt.plot(df['wl'], df['k'], 's', label='k Data')
    plt.plot(wl, interpolated[:, 0], '-', color='blue', label='Interpolated n')
    plt.plot(wl, interpolated[:, 1], '-', color='red', label='Interpolated k')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Value')
    plt.title('n and k vs Wavelength')
    plt.legend()
    plt.show()
