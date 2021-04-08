from math import sqrt, sin, cos, pi
import numpy as np
from constants import *
from stochastic import stochastic


def field(theta, phi, V_MTJ, n, NON, ENE, VNV):
    """
    Anisotropy module
    H_eff = H_PMA + H_D + H_TH + H_EX + H_VCMA, vectorwise
    Equations from paper section II E

    Input   : theta, phi, V_MTJ
    n       = for thermal noise generation
    NON     = 1 if want to add thermal disturbance, else 0
    ENE     = 1 if want to have exchange bias field, else 0
    VNV     = 1 if want to add VCMA (Voltage Controlled Magnetic Anisotropy) effect, else 0

    Output  : H_eff in x, y, z direction in array and H_eff_perpendicular
    """
    # directions in vectors
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    mx = sin(theta)*cos(phi)*ex
    my = sin(theta)*sin(phi)*ey
    mz = cos(theta)*ez
    m = np.add.reduce([mx, my, mz])  # m = mx + my + mz vectorwise

    # H_PMA
    H_PMA = 2*Ki/(u0*Ms*tf)*mz

    # H_VCMA
    H_VCMA = -2*beta*V_MTJ/(u0*Ms*tox*tf)*mz

    # Demagnetization field
    # H_D = -Ms(m.N), N is tensor(Nx 0 0, 0 Ny 0, 0 0 Nz)
    # Nx + Ny + Nz = 1, Nx = Ny by symmetry
    Nx = Ny = pi*tf/(4*D)
    Nz = 1-2*Nx
    N = np.array([Nx, 0, 0, 0, Ny, 0, 0, 0, Nz]).reshape(3, 3)
    H_D = -Ms*np.dot(m, N)

    # Exchange bias field
    Hx_ex = 0
    # can help to speed up switching, value = -50 Oe, 1 Oe = 1000/(4*pi) A/m
    Hy_ex = -50*1000/(4*pi)
    # see section III
    Hz_ex = 0
    H_EX = Hx_ex*ex + Hy_ex*ey + Hz_ex*ez

    # Thermal noise field
    H_th_mag = sqrt(2*kb*T*alpha/(u0*Ms*gamma*v*t_step))
    sigma = stochastic(n)
    H_TH = np.dot(H_th_mag, sigma)

    # total
    H_eff = H_PMA + H_D + NON*H_TH + ENE*H_EX + VNV*H_VCMA
    H_eff_perpendicular = np.dot(H_PMA + VNV*H_VCMA + H_D , ez) 
    return H_eff, H_eff_perpendicular
