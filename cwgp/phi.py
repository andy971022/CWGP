from autograd import elementwise_grad, grad
import autograd.numpy as np
import copy
from scipy.optimize import minimize

from cwgp.kernel import RBF, OU, Matern32
from cwgp.transformations import sal, sa, asinh, box_cox, inv_sal, inv_sa, inv_asinh, inv_box_cox


class Phi():
    FN_BANK = {
        "sal": {"fn": sal, "inv_fn": inv_sal, "par_len": 4},
        "sa": {"fn": sa, "inv_fn": inv_sa, "par_len": 2},
        "asinh": {"fn": asinh, "inv_fn": inv_asinh, "par_len": 4},
        "box_cox": {"fn": box_cox, "inv_fn": inv_box_cox, "par_len": 1},
    }

    KERNEL_BANK = {
        "OU": {"kern": OU, "params": 1, "init_scale": 1},
        "RBF": {"kern": RBF, "params": 1, "init_scale": 10},
        "Matern32": {"kern": Matern32, "params": 2, "init_scale": 10},
    }

    def __init__(
            self,
            fn,
            kernel="OU",
            kernel_params_estimate=False,
            par_len=None,
            n=1):
        self.fn = self.FN_BANK[fn]["fn"]  # a differentiable function
        self.inv_fn = self.FN_BANK[fn]["inv_fn"]
        self.d_fn = elementwise_grad(self.fn, 1)  # take derivative
        self.kernel = self.KERNEL_BANK[kernel]["kern"]
        self.kernel_params = self.KERNEL_BANK[kernel]["params"] if kernel_params_estimate else 0
        self.par_len = self.FN_BANK[fn]["par_len"]
        self.init_scale = self.KERNEL_BANK[kernel]["init_scale"]
        self.n = n

    def comp_phi(self, par, y):
        assert len(par) >= self.par_len * self.n, "Not enough parameters"
        comp = copy.deepcopy(y)
        d_comp = 1
        for i in range(0, self.n):
            d_comp *= self.d_fn(par[self.par_len * i:], comp)
            comp = self.fn(par[self.par_len * i:], comp)
        return comp, d_comp

    def inv_comp_phi(self, par, x):
        assert len(par) >= self.par_len * self.n, "Not enough parameters"
        if self.kernel_params:
            par = par[:-self.kernel_params]
        inv_comp = copy.deepcopy(x)
        for i in range(0, self.n):
            if i == 0:
                inv_comp = self.inv_fn(par[-self.par_len:], inv_comp)
            else:
                inv_comp = self.inv_fn(
                    par[-self.par_len * (i + 1):-self.par_len * i], inv_comp)
        return inv_comp

    def likelihood(self, par, y):
        phi_y, chain_d_sal = self.comp_phi(par, y)
        t_phi_y = np.transpose(phi_y)
        if self.kernel_params:
            cov_xx = self.kernel(phi_y, t_phi_y, par[-self.kernel_params:])
        else:
            cov_xx = self.kernel(phi_y, t_phi_y)
        gaussian_params = 0.5 * (t_phi_y) @ np.linalg.inv(cov_xx) @ phi_y
        return np.ravel(0.5 * np.log(np.linalg.det(cov_xx)) +
                        gaussian_params - sum(np.log(chain_d_sal)))

    def reml(self, method):
        pass

    def minimize_lf(self, y, method='l-bfgs-b', loop=True):
        # http://stackoverflow.com/questions/19843752/structure-of-inputs-to-scipy-minimize-function
        res = minimize
        res.success = False
        if loop:
            while res.success == False:
                try:
                    res = minimize(
                        self.likelihood,
                        self.init_scale * np.random.rand(
                            self.par_len *
                            self.n +
                            self.kernel_params),
                        args=(
                            y,
                        ),
                        method=method)
                except Exception:
                    pass
        self.res = res
        return res
