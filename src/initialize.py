from constants import *
from math import sqrt, exp


def init(PAP):
    """
    To initialize the VGSOT MTJ

    input:
    PAP = 1 if Parallel, 0 if Antiparallel

    output:
    R_MTJ, theta, mz, phi
    """

    if (PAP == 1):
        theta = pi-sqrt((kb*T)/(u0*Ms*Heff*v))
    elif (PAP == 0):
        theta = sqrt((kb*T)/(u0*Ms*Heff*v))
    phi = pi
    mz = cos(theta)

    # Brinkman resistance model
    F = (tox/(RA*phi_bar**(1/2))) * \
        exp((2*tox*(2*m*e*phi_bar)**(1/2))/h_bar)  # fitting factor
    # Magnetoresistance at parallel state
    Rp = (tox/(F*phi_bar**(1/2)*A1))*exp((2*tox*(2*m*e*phi_bar)**(1/2))/h_bar)
    # in Ohm, =100kOhm
    R_MTJ = Rp*(1+TMR/(TMR+2))/(1+TMR*cos(theta)/(TMR+2))
    return (R_MTJ, theta, mz, phi)