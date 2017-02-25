# -*- coding: utf-8 -*-
import numpy as np

def train_test_split(X, y, test_size=0.25):
    num = X.shape[0]
    test_size = int( num * test_size )
    test_index = np.random.choice(xrange(num), test_size)
    train_index = filter(lambda x: x not in test_index ,xrange(num))
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    return X_train, X_test, y_train, y_test
