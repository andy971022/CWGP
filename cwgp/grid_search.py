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
    transformations = params.get("transformations", ["sa", "box_cox"])
    params_combination = list(itertools.product(transformations, repeat=c))

    n_splits = kwargs.get("n_splits", 10)
    cv = kwargs.get("cv", False)
    shuffle = kwargs.get("shuffle", False)
    random_state = kwargs.get("random_state", None)

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
                cwgp_model = fit_transform(
                    param, y_train, x_train, **kwargs)
                cwgp[index][split_index] = estimator(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x=x,
                    y=t_data,
                    train=train,
                    val=val,
                    hyperparams=param,
                    cwgp_model=cwgp_model,
                    **kwargs)
        else:
            cwgp_model = fit_transform(
                param, t_data, x, **kwargs)
            cwgp[index]["result"] = estimator(
                x_train=x,
                y_train=t_data,
                cwgp_model=cwgp_model,
                hyperparams=param,
                **kwargs)
        cwgp[index]["cwgp_combination"] = param
    return cwgp


def fit_transform(param, y, t, **kwargs):
    cwgp_model = CWGP(list(param), **kwargs)
    cwgp_model.fit(y, t, **kwargs)
    y, y_d = cwgp_model.phi.comp_phi(
        cwgp_model.phi.res.x, y)
    return cwgp_model
