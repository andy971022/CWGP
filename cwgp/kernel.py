import numpy as np


class Kernel():
	def RBF(self, x, y, gamma=1):
		return np.exp(-gamma * np.abs(np.log(np.exp(x) @ np.exp(-(y)))))
