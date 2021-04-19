from constants import R_W

def electronic(V1, V2, V3, R_MTJ):
    """ 
    Represents the electronic module, equations from paper Section II B

    Input: 
    V1 = terminal voltage at MTJ
    V2 = terminal voltage at AFM strip
    V3 = right terminal voltage
    R_MTJ = MTJ resistance

    Output:
    I_SOT, V_MTJ as tuple
    """
    V_MTJ = R_MTJ * (4*V1 - 2*V2 - 2*V3)/(4*R_MTJ + R_W)
    I_SOT = (V2-V3)/R_W

    return (I_SOT, V_MTJ)