import autograd.numpy as np


def sal(par, y):
    return par[0] + par[1] * np.sinh(par[2] * np.arcsinh(y) + par[3])


def asinh(par, y):
    return par[0] + par[1] * np.arcsinh((y - par[2]) / par[3])


def box_cox(par, y):
    return (np.sign(y) * np.abs(y)**par[0] - 1) / par[0]


def sa(par, y):
    return np.sinh(par[0] * np.arcsinh(y) + par[1])
