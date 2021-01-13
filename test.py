import matplotlib.pyplot as plt
import autograd.numpy as np
import seaborn as sns
from cwgp.transformations import sal, box_cox, sa, asinh
from cwgp.cwgp import CWGP
import cwgp

print(cwgp)


betas = np.random.exponential(scale=5, size=100)
betas = betas[np.newaxis].reshape(-1, 1)


compgp = CWGP(sa, betas, transformations=4)

model = compgp.fit()

transformed_betas, d = compgp.phi.comp_phi(model.x)

sns.distplot(transformed_betas)
plt.show()
