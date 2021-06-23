import autograd.numpy as np


def sal(par, y):
    return par[0] + par[1] * np.sinh(par[2] * np.arcsinh(y) + par[3])


def inv_sal(par, x):
    return np.sinh((np.arcsinh((x - par[0]) / par[1]) - par[3]) / par[2])


def d_sal(par, y):
    return (par[1] * par[2] * np.cosh(par[2] * np.arcsinh(y) -
                                      par[3])) / (np.sqrt(1 + np.power(y, 2)))


def asinh(par, y):
    return par[0] + par[1] * np.arcsinh((y - par[2]) / par[3])


def inv_asinh(par, x):
    return (np.sinh((x - par[0]) / par[1]) * par[3]) + par[2]


def d_asinh(par, y):
    return (par[1]) / (np.power(par[3], 2) + np.power(y - par[2], 2))


def box_cox(par, y):
    return (np.sign(y) * np.abs(y)**par[0] - 1) / par[0]


def inv_box_cox(par, x):
    return np.sign(par[0] * x + 1) * (np.abs(par[0] * x + 1)**(1 / par[0]))


def d_box_cox(par, y):
    return np.power(np.abs(y), par[0] - 1)


def sa(par, y):
    return np.sinh(par[0] * np.arcsinh(y) + par[1])


def inv_sa(par, x):
    return np.sinh((np.arcsinh(x) - par[1]) / par[0])


def d_sa(par, y):
    return (par[0] * np.cosh(par[0] * np.arcsinh(y) - par[1])) / \
        (np.sqrt(1 + np.power(y, 2)))


def affine(par, y):
    return par[0] + par[1] * y


def inv_affine(par, x):
    return (x - par[0]) / par[1]


def d_affine(par, y):
    return par[1]
