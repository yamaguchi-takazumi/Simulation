import numpy as np
from scipy.special import ellipk, ellipe, roots_legendre

### Calculate the kernel of the integral that appears in the equivalent circuit model
def Kernel(r, z, rd, zd):
    dk  = 4.0 * r * rd
    dk /= (r + rd)**2 + (z - zd)**2
    ret = np.sqrt(dk) * (-2.0/dk * ellipe(dk) + (2.0/dk - 1.0) * ellipk(dk))
    return ret


### Calculate the kernel differentiated by rd.
def Kernel_diff(r, z, rd, zd):
    dk  = 4.0 * r * rd
    dk /= (r + rd)**2 + (z - zd)**2
    ellipk_ = ellipk(dk)
    ellipe_ = ellipe(dk)
    dk = np.sqrt(dk)
    dkdr  = 4.0 * r / ((r + rd)**2 + (z - zd)**2)
    dkdr -= 8.0 * r * rd * (r + rd) / ((r + rd)**2 + (z - zd)**2)**2
    ret  = -(ellipk_ - ellipe_) * dk**3 + ellipe_ / (2.0 * dk * (1.0 - dk**2))
    ret *=  dkdr
    return ret


### Calculate magneric flux density (Br / I, Bz / I) genereted by sheet current.
def MagFluxDensitySheet(r, z, rc, zc, height, n_gauss=10, rcomp=True, zcomp=True):
    dmu0 = 4.0e-7 * np.pi
    zmin, zmax = zc - height / 2.0, zc + height / 2.0
    
    if rcomp:
        if( r == 0.0):
            Br = 0.0
        else:
            Br  = Kernel(r, z, rc, zmax) - Kernel(r, z, rc, zmin)
            Br *= dmu0 * np.sqrt(rc / r) / (2.0 * np.pi * height)

    if zcomp:
        if( r == 0.0):
            Bz  = (zmax - z) / np.sqrt(rc**2 + (zmax - z)**2)
            Bz -= (zmin - z) / np.sqrt(rc**2 + (zmin - z)**2)
            Bz *= dmu0 / 2.0
        else:
            zd, wz = roots_legendre(n_gauss)
            zd = zmin * (1.0 - zd) / 2.0 + zmax * (1.0 + zd) / 2.0
        
            f  = Kernel(r, z, rc, zd) * np.sqrt(rc / r)
            f -= 2.0 * np.sqrt(r * rc) * Kernel_diff(r, z, rc, zd)
            Bz  = (f * wz).sum()
            Bz *= dmu0 / (8.0 * np.pi * r)
    
    if not zcomp:
        return Br
    
    if not rcomp:
        return Bz

    if rcomp and zcomp:
        return Br, Bz
    

### Calculate magneric flux density (Br / I, Bz / I) genereted by loop current.
def MagFluxDensityRect(r, z, rc, zc, width, thick, n_gauss=10, rcomp=True, zcomp=True):
    dmu0 = 4.0e-7 * np.pi
    rmin, rmax = rc - width / 2.0, rc + width / 2.0
    zmin, zmax = zc - thick / 2.0, zc + thick / 2.0
    
    if rcomp:
        if( r == 0.0):
            Br = 0.0
        else:
            rd, wr = roots_legendre(n_gauss)
            rd = rmin * (1.0 - rd) / 2.0 + rmax * (1.0 + rd) / 2.0

            f   = np.sqrt(rd / r) * (Kernel(r, z, rd, zmax) - Kernel(r, z, rd, zmin))
            Br  = (f * wr).sum()
            Br *= dmu0 / (4.0 * np.pi * thick)

    if zcomp:
        if( r == 0.0):
            Bz  = (zmax - z) / np.sqrt(rc**2 + (zmax - z)**2)
            Bz -= (zmin - z) / np.sqrt(rc**2 + (zmin - z)**2)
            Bz *= dmu0 / 2.0
        else:
            rd, wr = roots_legendre(n_gauss)
            zd, wz = roots_legendre(n_gauss)
            rd = rmin * (1.0 - rd) / 2.0 + rmax * (1.0 + rd) / 2.0
            zd = zmin * (1.0 - zd) / 2.0 + zmax * (1.0 + zd) / 2.0
            
            rd, zd = np.meshgrid(rd, zd)
            wr, wz = np.meshgrid(wr, wz)
        
            f  = Kernel(r, z, rd, zd) * np.sqrt(rd / r)
            f -= 2.0 * np.sqrt(r * rd) * Kernel_diff(r, z, rd, zd)
            Bz  = (f * wz * wr).sum()
            Bz *= dmu0 / (16.0 * np.pi * r)
    
    if not zcomp:
        return Br
    
    if not rcomp:
        return Bz

    if rcomp and zcomp:
        return Br, Bz


def ElectromagneticForce(I, Br, rad):
    return np.sum(2.0 * np.pi * rad * I * Br)
    

dmu0 = 4.0e-7 * np.pi

rcoil = 0.5
zcoil = 0.0
width = 1.0e-4
thick = 2.0e-4
height = 3.0e-4
current = 1.0

rp = rcoil * 1e-2
zp = zcoil + 2.0 * height

Br_cal, _ = MagFluxDensityRect(rcoil, zp, rcoil, zcoil, width, thick)
_, Bz_cal = MagFluxDensityRect(rp, zcoil, rcoil, zcoil, width, thick)
print("Calculated Velue  : {:.10e}, {:.10e}".format(Br_cal, Bz_cal))


Br_cal, _ = MagFluxDensitySheet(rcoil, zp, rcoil, zcoil, height)
_, Bz_cal = MagFluxDensitySheet(rp, zcoil, rcoil, zcoil, height)
Br_cal, Bz_cal = Br_cal * current, Bz_cal * current
print("Calculated Velue  : {:.10e}, {:.10e}".format(Br_cal, Bz_cal))


dist = np.abs(zp - zcoil)
Br_apx = dmu0 * current / (2.0 * np.pi * zp)
Bz_apx = dmu0 * current / (2.0 * rcoil)
print("Approximated Velue: {:.10e}, {:.10e}".format(Br_apx, Bz_apx))