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
            kernel_params_estimate=True,
            **kwargs
    ):
        self.fn = [self.FN_BANK[f]["fn"]
                   for f in fn]  # a differentiable function
        self.inv_fn = [self.FN_BANK[f]["inv_fn"] for f in fn[::-1]]
        self.par_len = [self.FN_BANK[f]["par_len"] for f in fn]
        self.d_fn = [elementwise_grad(f, 1)
                     for f in self.fn]  # take derivative
        self.kernel = self.KERNEL_BANK[kernel]["kern"]
        self.kernel_params = self.KERNEL_BANK[kernel]["params"] if kernel_params_estimate else 0
        self.init_scale = self.KERNEL_BANK[kernel]["init_scale"]

    def comp_phi(self, par, y):
        assert len(par) >= sum(self.par_len), "Not enough parameters"
        comp = copy.deepcopy(y)
        d_comp, par_len = 1, 0
        for i in range(0, len(self.fn)):
            d_comp *= self.d_fn[i](par[par_len:], comp)
            comp = self.fn[i](par[par_len:], comp)
            par_len += self.par_len[i]
        return comp, d_comp

    def inv_comp_phi(self, par, x):
        assert len(par) >= sum(self.par_len), "Not enough parameters"
        if self.kernel_params:
            par = par[:-self.kernel_params]
        inv_comp = copy.deepcopy(x)
        par_len = 0
        for i in range(0, len(self.fn)):
            par_len += self.par_len[::-1][i]
            inv_comp = self.inv_fn[i](par[-par_len:], inv_comp)
        return inv_comp

    def likelihood(self, par, y, t, mf):
        phi_y, chain_d_sal = self.comp_phi(par, y)
        t_phi_y = np.transpose(phi_y)
        t_t = np.transpose(t)
        mean_t = mf(t) if mf else np.zeros(t.shape)
        if self.kernel_params:
            cov_xx = self.kernel(t, t_t, par[-self.kernel_params:])
        else:
            cov_xx = self.kernel(t, t_t)
        gaussian_params = 0.5 * \
            (t_t - np.transpose(mean_t)) @ np.linalg.inv(cov_xx) @ (t - mean_t)
        return np.ravel(0.5 * np.log(np.linalg.det(cov_xx)) +
                        gaussian_params - sum(np.log(chain_d_sal)))

    def reml(self, method):
        pass

    def minimize_lf(
            self,
            y,
            t,
            mf=None,
            method='l-bfgs-b',
            loop=True,
            verbose=False,
            **kwargs):
        # http://stackoverflow.com/questions/19843752/structure-of-inputs-to-scipy-minimize-function
        res = minimize
        res.success = False
        if loop:
            while res.success == False:
                try:
                    res = minimize(
                        self.likelihood,
                        self.init_scale * np.random.rand(
                            sum(self.par_len) +
                            self.kernel_params),
                        args=(
                            y, t, mf
                        ),
                        method=method)
                except Exception as e:
                    if verbose:
                        print(e)
        self.res = res
        return res
