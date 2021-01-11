from cwgp.phi import Phi


class CWGP():
    def __init__(self, fn, data):
        self.phi = Phi(fn, data)

    def fit(self):
        return self.phi.minimize_lf()
