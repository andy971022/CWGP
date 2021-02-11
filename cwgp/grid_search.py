import itertools
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm

from cwgp.cwgp import CWGP


def grid_search(estimator, x, y, params={}, reverse_model_order=False):
    """
    Exhausts all given combinations of cwgp

    Parameters:
    ----------
    estimator: func
            a user defined function that uses x, the independent variable of the data, and the cwgp transformation of y,
            the dependent variable of the data.
x : np.array
    the independent variable of the data
y : np.array
    the dependent variable of the data, the target of the transformation
params : dict
    the range of grid search
    >>> {"c":2,"n":[1,2,3],"transformations":["sa","sal"]}
Returns
-------
cwpg : dict
    A dictionary containing all of the results
    """
    c = params.pop("c", 2)
    n = params.pop("n", [2])
    transformations = params.pop("transformations", ["sa", "sal"])
    cwgp_params = [transformations, n]
    params_product = list(itertools.product(*cwgp_params))
    print(params_product)
    params_combination = []

    for param in itertools.product(params_product, repeat=c):
        params_combination.append(param)

    cwgp = {}
    for index, param in enumerate(tqdm(params_combination)):
        print(param)
        t_data = copy.deepcopy(y)
        cwgp[index] = {"cwgp_combination": param}
        model_holder = []
        for t, d in param:
            cwgp_model = CWGP(t, n=d)
            cwgp_model.fit(t_data)
            t_data, t_data_d = cwgp_model.phi.comp_phi(
                cwgp_model.phi.res.x, t_data)
            model_holder.append(cwgp_model)
        if reverse_model_order:
            # For conveniences in inverse computation
            model_holder = model_holder[::-1]
        cwgp[index]["result"] = estimator(x, t_data, model_holder)
    return cwgp
