import cwgp

print(cwgp)

from cwgp.cwgp import CWGP
import seaborn as sns
import autograd.numpy as np
import matplotlib.pyplot as plt

betas = np.random.exponential(scale=5,size=100)
betas = betas[np.newaxis].reshape(-1,1)

def sal(par,y):
    return par[0]+par[1]*np.sinh(par[2]*np.arcsinh(y) + par[3])

compgp = CWGP(sal,betas)

model = compgp.fit()

transformed_betas, d = compgp.phi.comp_phi(model.x)

sns.distplot(transformed_betas)
plt.show()