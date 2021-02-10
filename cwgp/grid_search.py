import itertools
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm

from cwgp.cwgp import CWGP


def grid_search(estimator, x, y, params={}):
	c = params.pop("c",2)
	n = params.pop("n",[2])
	transformations = params.pop("transformations",["sa","sal"])

	cwgp_params = [transformations,n]

	params_product =  list(itertools.product(*cwgp_params))
	params_combination = []

	for param in itertools.permutations(params_product,c):
		if param not in params_combination:
			params_combination.append(param)

	cwgp = {}
	for index,param in tqdm(enumerate(params_combination)):
		t_data = copy.deepcopy(y)
		cwgp[index] = {"cwgp_combination":param}
		model_holder = []
		for t,d in param:
			cwgp_model = CWGP(t,n=d)
			cwgp_model.fit(t_data)
			t_data, t_data_d = cwgp_model.phi.comp_phi(cwgp_model.phi.res.x,t_data)
			model_holder.append(cwgp_model)
		cwgp[index]["result"] = estimator(x,t_data,model_holder)
	return cwgp