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


def estimator(**kwargs):
	print(kwargs)
	y_train = kwargs["y_train"]
	y_val = kwargs["y_val"]
	x_train = kwargs["x_train"]
	x_val = kwargs["x_val"]
	for cwgp_model in kwargs["model_holder"]:
		y_train, y_d = cwgp_model.phi.comp_phi(
            cwgp_model.phi.res.x, y_train)
		y_val, y_d = cwgp_model.phi.comp_phi(
			cwgp_model.phi.res.x, y_val)
	stats.probplot(y_train, dist="norm", plot=plt)
	stats.probplot(y_val, dist="norm", plot=plt)
	plt.show()

 # second param is a place holder
 # should give 9^3 combinations
grid_search(
    estimator, betas, betas, {
        "c": 3, "n": [
            1, 2, 3], "transformations": [
                "sa", "sal", "box_cox"]}, test="hi", cv=True,)
