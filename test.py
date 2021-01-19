import matplotlib.pyplot as plt
import autograd.numpy as np
import seaborn as sns
from cwgp.cwgp import CWGP
import cwgp

print(cwgp)


betas = np.random.exponential(scale=5, size=100)
betas = betas[np.newaxis].reshape(-1, 1)

sns.distplot(betas)
plt.show()

compgp = CWGP("sa", n=5)

model = compgp.fit(betas)

transformed_betas, d = compgp.phi.comp_phi(model.x, betas)

sns.distplot(transformed_betas)
plt.show()

inv_transformed_betas = compgp.phi.inv_comp_phi(model.x, transformed_betas)

fig, ax = plt.subplots(1, 2)
sns.distplot(inv_transformed_betas, ax=ax[0])
sns.distplot(betas, ax=ax[1])
plt.show()
