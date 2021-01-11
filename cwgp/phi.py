from autograd import elementwise_grad, grad
import autograd.numpy as np
import copy

from cwgp.kernel import Kernel


class Phi():
	def __init__(self, fn, data, kernel=Kernel.RBF, par_len=4, transformations=1):
		self.fn = fn  # a differentiable function
		self.d_fn = elementwise_grad(fn, 1)  # take derivative
		self.y = data
		self.kernel = kernel
		self.par_len = par_len
		self.n = transformations

	def comp_phi(self, par):
    	assert len(par) >= self.par_len * self.n, "Not enough Parameters"
		comp = copy.deepcopy(self.y)
		d_comp = 1
    	for i in range(0, n):
    		d_comp *= self.d_fn(par[self.par_len * i:], comp)
        	comp = self.fn(par[self.par_len * i:], comp)
    	return comp, dcomp

    def likelihood(self, par):
		phi_y, chain_d_sal = self.comp_phi(par,self.y)
		# phi_y = phi_y[np.newaxis].reshape(-1,1)
		t_phi_y = np.transpose(phi_y)

		# cov_xx = self.kernel(gamma,np.log(np.exp(phi_y) @ np.exp(-(t_phi_y))))
		cov_xx = self.kernel(phi_y,t_phi_y)
		gaussian_params = 0.5 * (t_phi_y) @ np.linalg.inv(cov_xx) @ phi_y

		return  0.5 * np.log(np.linalg.det(cov_xx)) + gaussian_params - sum(np.log(chain_d_sal))

	def minimize_lf(self, method = 'L-BFGS-B', loop=True):
		res = minimize(self.likelihood, np.random.rand(4*self.n), method='L-BFGS-B')
		if loop:
			while res.success == False:
				try:
					res = minimize(self.likelihood, np.random.rand(4*self.n), method='L-BFGS-B')
				except:
					pass
		return res
