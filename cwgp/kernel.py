import numpy as np


def OU(x, y, par=[1]):
    return np.exp(-par[0] *
                  np.abs(np.subtract.outer(x.reshape(-1), y.reshape(-1))))


def RBF(x, y, par=[1]):
    return np.exp(-par[0] *
                  np.abs(np.subtract.outer(x.reshape(-1), y.reshape(-1)))**2)
