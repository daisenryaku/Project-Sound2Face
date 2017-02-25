# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import expit

class LogisticClassifier(object):
  """
  逻辑回归二分类分类器
  """

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=10000,
            batch_size=200, verbose=False):
    num_train, dim = X.shape
    if self.W is None:
      self.W = 0.001 * np.random.randn(dim)
    loss_history = []
    for it in xrange(num_iters):
      index = np.random.choice(np.arange(num_train), batch_size)
      X_batch = X[index]
      y_batch = y[index]
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)
      self.W -= learning_rate * grad

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
    return loss_history

  def loss(self, X, y, reg):
    W = self.W
    num_train = X.shape[0]
    h = expit(X.dot(W))
    loss = np.sum(-y*np.log(h)-(1-y)*np.log(1-h))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW = np.dot(X.T,(h-y))
    dW /= num_train
    dW += reg*W
    return loss, dW

  def predict(self, X):
    y_pred = expit(np.dot(X, self.W))
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    return y_pred

  def score(self, X_test, y_test):
    """
    检验模型准确度
    Inputs:
    - X_test: 一个[num_test, D]的numpy数组，包含测试集数据
    - y_test: 一个[num_test, D]的numpy数组，包含测试集数据标记
    """
    y_test_pred = self.predict(X_test)
    num_test = y_test.shape[0]
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
