import numpy as np


def distance(x, y, metric="Euclidean"):
    if metric == "Euclidean":
        return np.abs(np.subtract.outer(x.reshape(-1), y.reshape(-1)))


def OU(x, y, par=[1]):
    return np.exp(-par[0] *
                  distance(x, y))


def RBF(x, y, par=[1]):
    return np.exp(-
                  np.power(distance(x / (par[0]), y / (par[0])), 2))


def Matern32(x, y, par=[1, 1]):
    return par[0] * (1 + np.sqrt(3) * distance(x, y) / par[1]) * \
        np.exp(-np.sqrt(3) * distance(x, y) / par[1])
