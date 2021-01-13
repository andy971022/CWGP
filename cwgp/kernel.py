import autograd.numpy as np


def OU(x, y, gamma=1):
    return np.exp(-gamma * np.abs(np.log(np.exp(x) @ np.exp(-(y)))))


def RBF(x, y, gamma=1):
    return np.exp(-gamma * np.abs(np.log(np.exp(x) @ np.exp(-(y))))**2)
