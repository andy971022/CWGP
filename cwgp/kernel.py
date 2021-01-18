import numpy as np


def OU(x, y, gamma=1):
    return np.exp(-gamma *
                  np.abs(np.subtract.outer(x.reshape(-1), y.reshape(-1))))


def RBF(x, y, gamma=1):
    return np.exp(-gamma *
                  np.abs(np.subtract.outer(x.reshape(-1), y.reshape(-1)))**2)
