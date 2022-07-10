import numpy as np
from scipy.special import ellipk, ellipe, roots_legendre

### Calculate the kernel of the integral that appears in the equivalent circuit model
def Kernel(r, z, rd, zd):
    dk  = 4.0 * r * rd
    dk /= (r + rd)**2 + (z - zd)**2
    ret = np.sqrt(dk) * (-2.0/dk * ellipe(dk) + (2.0/dk - 1.0) * ellipk(dk))
    return ret


### Caluclate the vector potential A / I of loop curret.
def VectorPotenstioalRect(r, z, rc, zc, width, thick, n_gauss=10):
    rmin, rmax = rc - width / 2.0, rc + width / 2.0
    zmin, zmax = zc - thick / 2.0, zc + thick / 2.0

    rd, wr = roots_legendre(n_gauss)
    zd, wz = rd, wr
    rd = rmin * (1.0 - rd) / 2.0 + rmax * (1.0 + rd) / 2.0
    zd = zmin * (1.0 - zd) / 2.0 + zmax * (1.0 + zd) / 2.0
    
    rd, zd = np.meshgrid(rd, zd)
    wr, wz = np.meshgrid(wr, wz)

    f = Kernel(r, z, rd, zd) * np.sqrt(rd / r)
    ret  = (f * wr * wz).sum()
    ret /= 8.0 * np.pi
    
    return ret


### Caluclate the vector potential A / I of sheet curret.
def VectorPotenstioalSheet(r, z, rc, zc, height, n_gauss=10):
    zmin, zmax = zc - height / 2.0, zc + height / 2.0

    zd, wz = roots_legendre(n_gauss)
    zd = zmin * (1.0 - zd) / 2.0 + zmax * (1.0 + zd) / 2.0
    
    f = Kernel(r, z, rc, zd) * np.sqrt(rc / r)
    ret  = (f * wz).sum()
    ret /= 4.0 * np.pi
    
    return ret


def InductanceRect(r, z, rc, zc, width, thick, n_gauss=10):
    dmu0 = 4.0e-7 * np.pi
    ret  = VectorPotenstioalRect(r, z, rc, zc, width, thick, n_gauss=n_gauss) * dmu0
    ret *= 2.0 * np.pi * r
    return ret


def InductanceSheet(r, z, rc, zc, height, n_gauss=10):
    dmu0 = 4.0e-7 * np.pi
    ret  = VectorPotenstioalSheet(r, z, rc, zc, height, n_gauss=n_gauss) * dmu0
    ret *= 2.0 * np.pi * r
    return ret    


rcoil = 1.0
width = 1.0e-4
zcoil = 1.0
thick = 2.0e-4
current = 1.0

rloop = 0.75
zloop = 1.5

dmu0 = 4.0e-7 * np.pi

L_cal = InductanceRect(rcoil, zcoil, rcoil, zcoil, width, thick)

M_cal = InductanceRect(rloop, zloop, rcoil, zcoil, width, thick)

M_cal_2 = InductanceSheet(rloop, zloop, rcoil, zcoil, width)

aval = width / 2.0
L_apx = dmu0 * rcoil * (np.log(8.0 * rcoil / aval) - 7.0 / 4.0)

dist = zcoil - zloop
M_apx = dmu0 * np.pi *  (rcoil*rloop)**2 / (2.0 * (rcoil**2 + dist**2)**(3/2))

print("Calculated Velue  : {:.10e}, {:.10e}, {:.10e}".format(L_cal, M_cal, M_cal_2))
print("Approximated Velue: {:.10e}, {:.10e}".format(L_apx, M_apx))