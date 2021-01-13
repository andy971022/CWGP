from autograd import elementwise_grad, grad
import autograd.numpy as np
import copy
from scipy.optimize import minimize

from cwgp.kernel import RBF, OU


class Phi():
    PAR_BANK = {
        "sal": 4,
        "sa": 2,
        "asinh": 4,
        "box_cox": 1,
    }

    def __init__(
            self,
            fn,
            data,
            kernel=OU,
            par_len=None,
            transformations=1):
        self.fn = fn  # a differentiable function
        self.d_fn = elementwise_grad(fn, 1)  # take derivative
        self.y = data
        self.kernel = kernel
        if par_len:
            self.par_len = par_len
        else:
            self.par_len = self.PAR_BANK[fn.__name__]
        self.n = transformations

    def comp_phi(self, par):
        assert len(par) >= self.par_len * self.n, "Not enough parameters"
        comp = copy.deepcopy(self.y)
        d_comp = 1
        for i in range(0, self.n):
            d_comp *= self.d_fn(par[self.par_len * i:], comp)
            comp = self.fn(par[self.par_len * i:], comp)
        return comp, d_comp

    def likelihood(self, par):
        phi_y, chain_d_sal = self.comp_phi(par)
        t_phi_y = np.transpose(phi_y)

        cov_xx = self.kernel(phi_y, t_phi_y)
        gaussian_params = 0.5 * (t_phi_y) @ np.linalg.inv(cov_xx) @ phi_y

        return np.ravel(0.5 * np.log(np.linalg.det(cov_xx)) +
                        gaussian_params - sum(np.log(chain_d_sal)))

    def minimize_lf(self, method='l-bfgs-b', loop=True):
        res = minimize(
            self.likelihood,
            np.random.rand(
                self.par_len * self.n),
            method=method)
        if loop:
            while res.success == False:
                try:
                    res = minimize(
                        self.likelihood, np.random.rand(
                            self.par_len * self.n), method=method)
                except BaseException:
                    pass
        return res
