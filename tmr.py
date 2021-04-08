from initialize import init
from constants import *

def tmr(V_MTJ, mz):
    """
    Tunnel Magnetoresistance module
    Input:
    V_MTJ, mz

    Output:
    R_MTJ
    """

    PAP = 0 #we want Parallel state
    R_p, _, _, _ = init(PAP)

    R_MTJ = R_p*(1+(V_MTJ/Vh)**2+TMR)/(1+(V_MTJ/Vh)**2+TMR*(0.5*(1+mz))) #because TMR_real = TMR/(1+V_MTJ^2/h^2)
    return R_MTJ
