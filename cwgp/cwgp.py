from autograd import elementwise_grad, grad
import autograd.numpy as np
from scipy.optimize import minimize
from kernel import Kernel
from phi import Phi
from likelihood import Likelihood

class CWGP():
	def __init__(self, fn, data):
		self.phi = Phi(fn)
		self.data = data

	def likelihood(self, fn,):
		comp, d_comp = self.phi.comp_phi(data)

	def minimize(self, lf):
		pass

