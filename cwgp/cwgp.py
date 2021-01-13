from cwgp.phi import Phi


class CWGP():
    def __init__(self, fn, data, *args, **kwargs):
        self.phi = Phi(fn, data, *args, **kwargs)

    def fit(self):
        return self.phi.minimize_lf()
