import autograd.numpy as np


def RBF(x, y, gamma=1):
    return np.exp(-gamma * np.abs(np.log(np.exp(x) @ np.exp(-(y)))))
