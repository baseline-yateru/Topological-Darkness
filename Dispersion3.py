import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

df = pd.read_csv('Data/Si_jaw.csv', sep = ',', names = ['eng','e1','e2'], skiprows = 3)
h = 4.14e-15  # Planck's constant in eV·s
c = 3e8  # Speed of light in m/s


def local_interpolator(df, z_col='eng', x_col='e1', y_col='e2'):
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
        eng = h*c/z_val
        n = np.sqrt(interp_x(eng)+ 1j* interp_y(eng))
        return np.column_stack((np.real(n), np.imag(n)))
    return interpolator

def plotter():
    eng = np.linspace(0.2, 6.6, 1000)  # Wavelength range in nm
    wl = h*c/eng  # Energy in eV
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    e1 = np.array(df['e1'].values)
    e2 = np.array(df['e2'].values)
    data_wl = h*c/np.array(df['eng'])

    n = np.sqrt(e1 + 1j*e2)
    ax.scatter(np.real(n), np.imag(n), data_wl, label='Data Points')

    #ax.scatter(df['e1'], df['e2'], h*c/np.array(df['eng']), label='Data Points')
    dis = local_interpolator(df)(eng)
    #print((dis[:,0] + 1j*dis[:,1])[-6:-1])
    #print(ref[-6:-1])
    #ax.plot(dis[:, 0], dis[:, 1], wl, color='red', label='Permitivity')
    ax.plot(dis[:, 0], dis[:, 1], wl, color='red', label='Refractive Index')
    ax.set_xlabel('n')
    ax.set_ylabel('k')
    ax.set_zlabel('Wavelength (nm)')
    ax.set_title('3D Plot of n, k vs Wavelength')
    ax.legend()
    plt.show()