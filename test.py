import matplotlib.pyplot as plt
import autograd.numpy as np
import seaborn as sns
from scipy import stats


from cwgp.cwgp import CWGP
from cwgp.grid_search import grid_search
import cwgp

print(cwgp)
np.random.seed(seed=32)
SIZE = 100

betas = np.random.exponential(scale=5, size=SIZE)

sns.distplot(betas)
plt.show()

compgp = CWGP(["box_cox", "sal", "sal","sa"], kernel="OU", kernel_params_estimate=True)

model = compgp.fit(betas, np.arange(SIZE, dtype="float"), verbose=True)
print(compgp.phi.res.x)

transformed_betas, d = compgp.phi.comp_phi(model.x, betas)

sns.distplot(transformed_betas)
plt.show()
plt.plot(np.arange(SIZE, dtype="float"), betas)
plt.show()
stats.probplot(transformed_betas, dist="norm", plot=plt)
plt.show()
plt.plot(np.arange(SIZE, dtype="float"), transformed_betas)
plt.show()


print(model.x)
inv_transformed_betas = compgp.phi.inv_comp_phi(model.x, transformed_betas)

fig, ax = plt.subplots(1, 2)
sns.distplot(inv_transformed_betas, ax=ax[0])
sns.distplot(betas, ax=ax[1])
plt.show()


def estimator(**kwargs):
    if kwargs.get("cv", False):
        y_train = kwargs["y_train"]
        y_val = kwargs["y_val"]
        x_train = kwargs["x_train"]
        x_val = kwargs["x_val"]
        cwgp_model = kwargs["cwgp_model"]

        y_train, y_d = cwgp_model.phi.comp_phi(
            cwgp_model.phi.res.x, y_train)
        y_val, y_d = cwgp_model.phi.comp_phi(
            cwgp_model.phi.res.x, y_val)
        sns.distplot(y_train)
        plt.show()
        # stats.probplot(y_train, dist="norm", plot=plt)
        sns.distplot(y_val)
        # stats.probplot(y_val, dist="norm", plot=plt)
        plt.show()


 # second param is a place holder
 # should give 9^3 combinations
# grid_search(
#     estimator, betas, np.arange(SIZE, dtype="float"), {
#         "c": 1, "transformations": [
#                 "sa"]}, test="hi")


grid_search(
    estimator, np.arange(SIZE , dtype="float"), betas,  {
        "c": 4, "transformations": [
                "box_cox", "sa", "sal"]}, test="hi", cv=True, n_splits=3)
