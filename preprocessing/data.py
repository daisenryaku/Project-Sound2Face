import numpy as np

class StandardScaler(object):
    def __init__(self):
        pass
    def fit_transform(self, X):
        mean = np.mean(X)
        std = np.std(X)
        X -= mean
        X /= std
        return X
