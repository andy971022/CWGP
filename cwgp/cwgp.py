from cwgp.phi import Phi


class CWGP():
    def __init__(self, fn, *args, **kwargs):
        self.phi = Phi(fn, *args, **kwargs)

    def fit(self, y, t, **kwargs):
        return self.phi.minimize_lf(y, t, **kwargs)

    def normality_test(self):
        pass
