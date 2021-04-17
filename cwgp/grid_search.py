import itertools
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from sklearn.model_selection import KFold

from cwgp.cwgp import CWGP


def grid_search(
        estimator,
        x,
        y,
        params={},
        **kwargs):
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
    c = params.get("c", 2)
    n = params.get("n", [2])
    transformations = params.get("transformations", ["sa", "sal"])
    cwgp_params = [transformations, n]
    params_product = list(itertools.product(*cwgp_params))
    params_combination = list(itertools.product(params_product, repeat=c))
    print(params_product)

    n_splits = kwargs.get("n_splits", 10)
    cv = kwargs.get("cv", False)
    shuffle = kwargs.get("shuffle", False)
    random_state = kwargs.get("random_state", 42)
    reverse_model_order = kwargs.get("reverse_model_order", False)

    cwgp = {}
    for index, param in enumerate(tqdm(params_combination)):
        cwgp[index] = {}

        t_data = copy.deepcopy(y)
        if cv:
            kf = KFold(
                n_splits=n_splits,
                random_state=random_state,
                shuffle=shuffle)
            for split_index, (train, val) in enumerate(kf.split(t_data)):
                x_train, x_val = x[train], x[val]
                y_train, y_val = t_data[train], t_data[val]
                model_holder = fit_transform(
                    param, y_train, reverse_model_order=reverse_model_order)
                cwgp[index][split_index] = estimator(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    hyperparams=param,
                    model_holder=model_holder,
                    **kwargs)
        else:
            model_holder = fit_transform(
                param, t_data, reverse_model_order=reverse_model_order)
            cwgp[index]["result"] = estimator(
                x_train=x,
                y_train=t_data,
                model_holder=model_holder,
                hyperparams=param,
                **kwargs)
        cwgp[index]["cwgp_combination"] = param
    return cwgp


def fit_transform(param, y, reverse_model_order=False):
    model_holder = []
    for t, d in param:
        cwgp_model = CWGP(t, n=d)
        cwgp_model.fit(y)
        y, y_d = cwgp_model.phi.comp_phi(
            cwgp_model.phi.res.x, y)
        model_holder.append(cwgp_model)
    if reverse_model_order:
        # For conveniences in inverse computation
        model_holder = model_holder[::-1]
    return model_holder
