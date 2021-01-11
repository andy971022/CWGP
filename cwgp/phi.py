from autograd import elementwise_grad, grad
import autograd.numpy as np
import copy

class Phi():
	def __init__(self, fn, par_len=4, transformations=1): #
		self.fn = fn # a differentiable function
		self.d_fn = elementwise_grad(sal,1) # take derivative
		self.par_len = par_len
		self.n = transformations

	def comp_phi(self, par, y):
		comp = copy.deepcopy(y)
		d_comp = 1
    	for i in range(0,n):
    		d_comp *= self.d_fn(par[self.par_len*i:],comp)
        	comp = self.fn(par[self.par_len*i:],comp)
    	return comp, dcomp

    def likelihood(self, par, y):
		phi_y, chain_d_sal = self.comp_phi(par,y)
		# phi_y = phi_y[np.newaxis].reshape(-1,1)
		t_phi_y = np.transpose(phi_y)

		cov_xx = cov_kernel(gamma,np.log(np.exp(phi_y) @ np.exp(-(t_phi_y))))
		gaussian_params = 0.5 * (t_phi_y) @ np.linalg.inv(cov_xx) @ phi_y


		return  0.5 * np.log(np.linalg.det(cov_xx)) + gaussian_params - sum(np.log(chain_d_sal))