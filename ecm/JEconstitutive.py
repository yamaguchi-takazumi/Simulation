import numpy as np
import matplotlib.pyplot as plt

# Calculate electric field E from currnt density j by using Ohm's law.
def OhmsLaw(j, sigma):
    return sigma * j


# Calculate electric field E from currnt density j by using the power law.
def PowerLaw(j, jc, Ec=1.0e-3, n_value=20):
    ret  = np.power(np.abs(j) / jc, n_value)
    ret *= Ec * np.sign(j)
    return ret


# Calculate electric field E from currnt density j by using the power law.
def PowerLaw(j, jc, Ec=1.0e-3, n_value=20):
    ret  = np.power(np.abs(j) / jc, n_value)
    ret *= Ec * np.sign(j)
    return ret


# Calculate electric field E from currnt density j by using the flow-creap model.
def FlowCreapModel(j, jc, T, Ec=1.0e-3, rho=7.62e-10, U0=9.6e-2):
    kB = 1.380649e-23
    eV = 1.602176e-19	
    coef = U0*eV / (kB * T)
    
    le_ret = Ec * np.sinh(coef * j / jc) / np.sinh(coef)
    gt_ret = Ec + rho * (j - jc)
    
    j_le_jc = j < jc
    return j_le_jc * le_ret + (1 - j_le_jc) * gt_ret


# Calculate electric field E from currnt density j by using the flow-creap model.
def CriticalStateModel(j, jc, Emax=1.0e+32):
    j_le_jc = j < jc
    return (1 - j_le_jc) * Emax


# Convert from electric field E to voltage V.
def ConvE2V(E, rad):
    return 2.0 * np.pi * rad * E


# Convert from current I to current density j.
def ConvI2j(I, area):
    return I / area


x = np.linspace(0.0, 1.5, 100)

fig, ax = plt.subplots()

ax.plot(x, OhmsLaw(x, 2.0), label="Ohm's law")
ax.plot(x, CriticalStateModel(x, 1.0), label="critical state model")
ax.plot(x, PowerLaw(x, 1.0, Ec=1.0), label="power law")
ax.plot(x, FlowCreapModel(x, 1.0, 77, Ec=1.0), label="flow-creap model")

ax.set_xlabel("Current Density, J / Jc")
ax.set_ylabel("Electric Filed, E / Ec")
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)
ax.legend()

plt.show()