import numpy as np

def custom_disp(p0, p1):
    x0, y0, z0 = p0
    x1, y1, z1 = p1

    dz = z1 - z0
    if dz == 0:
        raise ValueError("Line is parallel to XY plane; cannot parametrize with respect to z")
    def line(z):
    
        x = x0 + (x1 - x0) * (z - z0) / dz
        y = y0 + (y1 - y0) * (z - z0) / dz
        return x + 1j*y
    return line

def vect(p0, p1):
    return np.array(p0) - np.array(p1)