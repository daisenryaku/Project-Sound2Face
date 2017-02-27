# -*- coding: utf-8 -*-
import csv
import numpy as np
from os.path import dirname
from os.path import join

class Bunch(dict):
    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass

def load_data(module_path, data_file_name):
    with open(join(module_path, 'data', data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])

        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target

def load_breast_cancer():
    module_path = dirname(__file__)
    data, target = load_data(module_path, 'breast_cancer.csv')
    return Bunch(data=data, target=target)

def load_features():
    module_path = dirname(__file__)
    data, target = load_data(module_path, 'features.csv')
    return Bunch(data=data, target=target)
