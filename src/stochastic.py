import numpy as np

def stochastic(n):
    """
    A helper function for the thermal noise, useful in H_TH in Anisotropy module
    Return Gaussian random number

    Input: n as standard deviation
    Output: [sigmaX, sigmaY, sigmaZ]
    """

    sigma = np.random.normal(0, n, 3)  # outputs array, 1x3
    normCoeff = np.sqrt(np.sum(sigma*sigma))
    sigma /= normCoeff
    assert(np.sqrt(np.sum(sigma*sigma)) - 1.00 < 1e-15)  # assert normalized
    return sigma