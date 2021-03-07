import matplotlib.pyplot as plt
import autograd.numpy as np
import seaborn as sns
from scipy import stats


from cwgp.cwgp import CWGP
from cwgp.grid_search import grid_search
import cwgp

print(cwgp)


betas = np.random.exponential(scale=5, size=100)

sns.distplot(betas)
plt.show()

compgp = CWGP("box_cox", n=1, kernel="OU", kernel_params_estimate=False)

model = compgp.fit(betas, method="l-bfgs-b")

transformed_betas, d = compgp.phi.comp_phi(model.x, betas)

sns.distplot(transformed_betas)
plt.show()

stats.probplot(transformed_betas, dist="norm", plot=plt)
plt.show()


print(model.x)
inv_transformed_betas = compgp.phi.inv_comp_phi(model.x, transformed_betas)

fig, ax = plt.subplots(1, 2)
sns.distplot(inv_transformed_betas, ax=ax[0])
sns.distplot(betas, ax=ax[1])
plt.show()


def estimator(*args,**kwargs):
    print(kwargs["test"])
    pass


 # second param is a place holder
 # should give 9^3 combinations
grid_search(
    estimator, betas, betas, {
        "c": 3, "n": [
            1, 2, 3], "transformations": [
                "sa", "sal", "box_cox"]}, test="hi")
