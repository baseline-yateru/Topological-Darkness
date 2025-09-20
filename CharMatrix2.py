import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

def sqrt_c(x):
    """
    I actually don't know why this works, but it does for most cases, probably needs correction.
    """
    if np.imag(x) < 0:
        return np.conj(np.sqrt(x))
    else:
        return np.sqrt(x)

class ArrayDict():
    """
    A class that associates two NumPy arrays (`arr_s` and `arr_p`) with each instance.
    Attributes:
        arr_s (np.ndarray): The first associated NumPy array.
        arr_p (np.ndarray): The second associated NumPy array.
    Methods:
        __init__(arr_s: np.ndarray, arr_p: np.ndarray):
            Initializes the ArrayDict with two NumPy arrays.
        __mul__(other):
            Returns a new ArrayDict whose arrays are the matrix product of the corresponding arrays
            from self and other.
    """

    def __init__(self, arr_s: np.ndarray, arr_p: np.ndarray):
        super().__init__()
        self.arr_s = arr_s
        self.arr_p = arr_p

    def __mul__(self, other):
        return ArrayDict(self.arr_s @ other.arr_s, self.arr_p @ other.arr_p)
    
class CharMatrix(np.ndarray):
    """
    CharMatrix(ang_i: complex, n_i: complex, n_r: complex, d: float, wl: float)
    A subclass of numpy.ndarray representing the characteristic matrix for a thin film layer in optics, 
    for both s- and p-polarizations.
    Parameters
    ----------
    ang_i : float
        Incident angle (in radians). !Always from the TOP LAYER!
    n_i : complex
        Refractive index of the incident medium. !Always from the TOP LAYER!
    n_r : complex
        Refractive index of the thin film layer.
    d : float
        Thickness of the thin film layer.
    wl : float
        Wavelength of the incident light.
    Attributes
    ----------
    arr_s : CharMatrix
        Characteristic matrix for s-polarized light.
    arr_p : CharMatrix
        Characteristic matrix for p-polarized light.
    ang_i : float
        Incident angle. !Always from the TOP LAYER!
    ang_r : complex
        Refraction angle inside the current layer.
    n_i : complex
        Refractive index of the incident medium. !Always from the TOP LAYER!
    n_r : complex
        Refractive index of the current layer.
    d : float
        Thickness of the thin film layer.
    wl : float
        Wavelength of the incident light.
    phi : complex
        Phase thickness of the layer.
    Notes
    -----
    The characteristic matrices are computed using the transfer matrix method for thin films, 
    with separate matrices for s- and p-polarizations.
    """
    def __new__(cls, ang_i: float, n_i: complex, n_r: complex, d: float, wl: float):
        # Define Parameters
        ang_i = np.float64(ang_i)
        n_i = np.complex128(n_i)
        n_r = np.complex128(n_r)
        d = np.float64(d)
        wl = np.float64(wl)
        cos_ang_r = sqrt_c(1-(n_i/n_r * np.sin(ang_i))**2) # Replaced Snell's Law to calculate cos(ang_r) directly
        phi = (2 * d * n_r * np.pi * cos_ang_r / wl).astype(np.complex128)
        cos_phi = np.cos(phi).astype(np.complex128)
        sin_phi = np.sin(phi).astype(np.complex128)
        p = n_r*cos_ang_r
        q = 1/n_r * cos_ang_r

        # Define Characteristic Matrices
        arr_s = np.array([
            [cos_phi, -1j/p * sin_phi],
            [-1j * p * sin_phi, cos_phi]
        ], dtype=np.complex128)

        arr_p = np.array([
            [cos_phi, -1j/q * sin_phi ],
            [-1j * q * sin_phi, cos_phi]
        ], dtype=np.complex128)

        # Create Object and Define Attributes
        obj = super().__new__(cls, shape=arr_s.shape, dtype=np.complex128)
        obj.arr_s = arr_s.view(cls)
        obj.arr_p = arr_p.view(cls)
        obj.ang_i = ang_i
        obj.ang_r = np.arccos(cos_ang_r)
        obj.n_i = n_i
        obj.n_r = n_r
        obj.d = d
        obj.wl = wl
        obj.phi = phi
        return obj

class Substrate():
    """
    Substrate represents a multilayer optical substrate using the characteristic matrix method.
    Attributes:
        matls (list): List of CharMatrix objects for each layer.
        ang_i (float): Incident angle (in radians) from the top layer.
        n (np.array): Array of refractive indices for each medium (including incident and exit media).
        d (np.array): Array of thicknesses for each layer (not including incident and exit media).
        wl (float): Wavelength of the incident light.
        ad (ArrayDict): Combined characteristic matrix for the entire stack.
    Methods:
        ref_s():
            Calculates the reflection coefficient for s-polarized (TE) light.
                complex or float: Reflection coefficient for s-polarization.
        ref_p():
            Calculates the reflection coefficient for p-polarized (TM) light.
                complex or float: Reflection coefficient for p-polarization.
        tran_s():
            Calculates the transmission coefficient for s-polarized (TE) light.
                complex or float: Transmission coefficient for s-polarization.
        tran_p():
            Calculates the transmission coefficient for p-polarized (TM) light.
                complex or float: Transmission coefficient for p-polarization.
        elip():
            Calculates the complex ellipsometric ratio (rho = r_p / r_s).
                complex or float: Ellipsometric ratio.
    """
    def __init__(self, ang_i: float, n: np.array, d: np.array, wl: float):
        #Check if there are expected number of layers, as ambient layers don't have thickness
        if len(n) == len(d) + 2:               
            matls = []
            for i, di in enumerate(d):
                # Find the Characteristic Matrices for each layer
                mat = CharMatrix(ang_i, n[0], n[i+1], di, wl)
                matls.append(mat)
            self.cos_ang_i = np.cos(ang_i)
            self.cos_ang_r = sqrt_c(1 - (n[0] / n[-1])**2 * np.sin(ang_i)**2)
            
        else:
            raise ValueError("Number of refractive indices must be two more than the number of layers.")
        
        ad = ArrayDict(np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]))
        for i, mat in enumerate(matls):
            # Calculate the final Cahracteristic Matrix
            ad *= ArrayDict(mat.arr_s, mat.arr_p)
        
        self.matls = matls
        self.n = n
        self.d = d
        self.wl = wl
        self.ad = ad

    #Trivial         
    def ref_s(self):        
        n1 = self.n[0]
        nl = self.n[-1]
        cos_a1 = self.cos_ang_i
        cos_al = self.cos_ang_r

        p1 = cos_a1 * n1
        pl = cos_al * nl

        arr_s = self.ad.arr_s

        R =  p1 * (arr_s[0, 1] * pl + arr_s[0, 0]) -  (arr_s[1,0] + arr_s[1,1] * pl)
        A =  p1 * (arr_s[0, 1] * pl + arr_s[0, 0]) +  (arr_s[1,0] + arr_s[1,1] * pl)
        return R / A
        
    def ref_p(self):
        n1 = self.n[0]
        nl = self.n[-1]
        cos_a1 = self.cos_ang_i
        cos_al = self.cos_ang_r

        q1 = cos_a1 / n1
        ql = cos_al / nl

        arr_p = self.ad.arr_p

        R =  q1 * (arr_p[0, 1] * ql + arr_p[0, 0]) -  (arr_p[1,0] + arr_p[1,1] * ql)
        A =  q1 * (arr_p[0, 1] * ql + arr_p[0, 0]) +  (arr_p[1,0] + arr_p[1,1] * ql)
        return R / A

    def tran_s(self):
        n1 = self.n[0]
        nl = self.n[-1]
        cos_a1 = self.cos_ang_i
        cos_al = self.cos_ang_r

        p1 = cos_a1 * n1
        pl = cos_al * nl

        arr_s = self.ad.arr_s

        A =  p1 * (arr_s[0, 1] * pl + arr_s[0, 0]) +  (arr_s[1,0] + arr_s[1,1] * pl)
        return 2 * p1 / A
        
    def tran_p(self):
        n1 = self.n[0]
        nl = self.n[-1]
        cos_a1 = self.cos_ang_i
        cos_al = self.cos_ang_r

        q1 = cos_a1 / n1
        ql = cos_al / nl

        arr_p = self.ad.arr_p

        A =  q1 * (arr_p[0, 1] * ql + arr_p[0, 0]) +  (arr_p[1,0] + arr_p[1,1] * ql)
        return 2*q1 / A
    
    def elip(self):
        return self.ref_p() / self.ref_s()
"""
ang = np.linspace(0, np.pi/2, 10000)

M = list(map(lambda x: Substrate(x, np.array([1, 1.5, 1.5]), np.array([0]), 500e-9), ang))

rs_arr = np.array([m.ref_s() for m in M])
rp_arr = np.array([m.ref_p() for m in M])
el_arr = np.array([m.elip() for m in M])
#print([m.angls for m in M])
plt.plot(ang, np.abs(rs_arr)**2, label='Reflectance s-polarized')
plt.plot(ang, np.abs(rp_arr)**2, label='Reflectance p-polarized')
plt.plot(ang, np.abs(el_arr), label=r'Ellipsometric Ratio $\rho = r_p / r_s$')
#plt.plot(ang, np.angle(el_arr), label=r'Ellipsometric Ratio $\rho = r_p / r_s$')

plt.xlabel('Incident Angle (radians)')
plt.ylabel(r'$\Psi$, $R_{s}$ & $R_{p}$')
plt.title(r'$\Psi$, $R_{s}$ & $R_{p}$ vs Incident Angle')
plt.legend()
plt.grid()
#plt.savefig("Figure_1.png", dpi=600)
plt.show()
"""
