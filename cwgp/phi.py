from autograd import elementwise_grad, grad
import autograd.numpy as np
import copy
from scipy.optimize import minimize

from cwgp.kernel import RBF, OU
from cwgp.transformations import sal, sa, asinh, box_cox, inv_sal, inv_sa, inv_asinh, inv_box_cox


class Phi():
    FN_BANK = {
        "sal": {"fn": sal, "inv_fn": inv_sal, "par_len": 4},
        "sa": {"fn": sa, "inv_fn": inv_sa, "par_len": 2},
        "asinh": {"fn": asinh, "inv_fn": inv_asinh, "par_len": 4},
        "box_cox": {"fn": box_cox, "inv_fn": inv_box_cox, "par_len": 1},
    }

    def __init__(
            self,
            fn,
            data,
            kernel=OU,
            par_len=None,
            transformations=1):
        self.fn = self.FN_BANK[fn]["fn"]  # a differentiable function
        self.inv_fn = self.FN_BANK[fn]["inv_fn"]
        self.d_fn = elementwise_grad(self.fn, 1)  # take derivative
        self.y = data
        self.kernel = kernel
        if par_len:
            self.par_len = par_len
        else:
            self.par_len = self.FN_BANK[fn]["par_len"]
        self.n = transformations

    def comp_phi(self, par):
        assert len(par) >= self.par_len * self.n, "Not enough parameters"
        comp = copy.deepcopy(self.y)
        d_comp = 1
        for i in range(0, self.n):
            d_comp *= self.d_fn(par[self.par_len * i:], comp)
            comp = self.fn(par[self.par_len * i:], comp)
        return comp, d_comp

    def inv_comp_phi(self, par, x):
        assert len(par) >= self.par_len * self.n, "Not enough parameters"
        inv_comp = copy.deepcopy(x)
        for i in range(0, self.n):
            if i == 0:
                inv_comp = self.inv_fn(par[-self.par_len:], inv_comp)
            else:
                inv_comp = self.inv_fn(
                    par[-self.par_len * (i + 1):-self.par_len * i], inv_comp)
        return inv_comp

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
