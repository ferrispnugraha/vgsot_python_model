from constants import *
from anisotropy import field
from math import cos, sin


def switching(V_MTJ, I_SOT, R_MTJ, theta, phi, ESTT, ESOT, VNV = 1, NON=0, R_SOT_FL_DL=0.83, R_STT_FL_DL = 0):
    """
    Dynamic-switching module and solving LLG analytically
    Input: V_MTJ, I_SOT, R_MTJ, theta, phi
    ESTT = 1 for STT effect, else 0
    ESOT = 1 for SOT effect, else 0
    VNV =  1 to reproduce manual, 0 for some in paper
    NON = 0 to reproduce manual and some in paper
    R_SOT_FL_DL = 0.83 to reproduce manual, 0 for paper

    Output:
    mz (from theta_1)
    phi_1, and theta_1
    => finding new angles after 1 time step
    """

    # for Anisotropy field module
    n = 1
    ENE = 1
    # NON = 0
    # VNV = 0 #0,to reproduce paper, 1 to reproduce manual

    H_EFF, _ = field(theta, phi, V_MTJ, n, NON, ENE, VNV)

    # Prepare parameters for LLG solving
    I_MTJ = V_MTJ/R_MTJ
    J_STT = I_MTJ/A1
    J_SOT = I_SOT/A2

    # Reduced gyromagnetic ratio
    gamma_red = gamma/(1+alpha**2)
    # see section II D
    H_DL_STT = ESTT * h_bar * P * J_STT/(2*e*u0*Ms*tf)         # Damping-Like for STT, ESTT has impact
    H_FL_STT = R_STT_FL_DL * H_DL_STT                          # Field-Like for STT
    H_DL_SOT = ESOT * h_bar * theta_SH * J_SOT/(2*e*u0*Ms*tf)  # Damping-Like for SOT, ESOT has impact
    H_FL_SOT = R_SOT_FL_DL * H_DL_SOT                          # Field-Like for SOT

    # Landau-Lifshitz-Gilbert equation solving
    # use equation 9 section II D
    # m = mx ex + my ey + mz ez
    # m_p = (0, 0, 1) = electron polarization direction of the spin polarized current induced by STT
    # m_s = (-1, 0, 0) = pure spin current induced by spin-orbit coupling
    # to create D 6
    dtheta_dt = gamma_red*(
        H_EFF[0]*(alpha*cos(theta)*cos(phi) - sin(phi))
        + H_EFF[1]*(alpha*cos(theta)*sin(phi) + cos(phi))
        - H_EFF[2]*(alpha*sin(theta))
        + sin(theta)*(alpha*H_FL_STT - H_DL_STT)
        - H_DL_SOT*(alpha*sin(phi) + cos(theta)*cos(phi))
        + H_FL_SOT*(alpha*cos(theta)*cos(phi)-sin(phi))
    )

    dphi_dt = gamma_red*(
        1/sin(theta)*(
            H_EFF[0]*(-alpha*sin(phi) - cos(theta)*cos(phi))
            + H_EFF[1]*(alpha*cos(phi)-cos(theta)*sin(phi))
            + H_EFF[2]*sin(theta)
        )
        - (alpha*H_DL_STT + H_FL_STT)
        - H_DL_SOT/sin(theta)*(alpha*cos(theta)*cos(phi)-sin(phi))
        - H_FL_SOT/sin(theta)*(alpha*sin(phi)-cos(theta)*cos(phi))
    )

    theta_1 = theta + dtheta_dt*t_step
    phi_1 = phi + dphi_dt*t_step
    mz = cos(theta_1)
    return (mz, phi_1, theta_1)