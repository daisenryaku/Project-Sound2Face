# -*- coding: utf-8 -*-
import numpy as np

class LinearClassifier(object):
  """
  线性回归分类器, 基于svm loss function
  """

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    使用随机梯度下降训练线性回归分类器

    Inputs:
    - X：一个[num_train, D]的numpy数组; num_train:测试样本个数，D：样本元素维度
    - y: 一个[num_train,]的numpy数组，包含训练样本标记; y[i] = c 意味着 X[i] 标记为c. 0 <= c < C
      (假设一共有 k 类，y 取值从0到k-1)
    - learning_rate: (float) 学习率
    - reg: (float) 正则化长度
    - num_iters: (integer) 步长
    - batch_size: (integer) 每一步训练使用的样本数
    - verbose: (boolean) 如果为真，打印训练时每一步的loss值

    Outputs:
    - loss_history 一个包含每一步loss值的列表
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1
    if self.W is None:
      self.W = 0.001 * np.random.randn(dim, num_classes)

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

  def predict(self, X):
    """
    对测试集产生预测输出

    Inputs:
    - X：一个[num_test, D]的numpy数组; num_test:测试样本个数，D：样本元素维度

    Returns:
    - y: 一个[num_test,]的numpy数组，包含对x的预测输出标记
    """
    y_pred = np.zeros(X.shape[1])
    y_pred = np.argmax( np.dot(X, self.W) ,axis=1)
    return y_pred

  def loss(self, X, y, reg):
    """
    计算损失函数及其导数

    Inputs:
    - X：一个[num_train, D]的numpy数组; num_train:测试样本个数，D：样本元素维度
    - y: 一个[num_train,]的numpy数组，包含训练样本标记;

    Returns:
    - loss: (single float)损失函数值
    - dW: 梯度
    """
    W = self.W
    loss = 0.0
    dW = np.zeros(W.shape)
    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_score = scores[np.arange(num_train), y]
    margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1.0)
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    X_mask = np.zeros(margins.shape)
    X_mask[margins > 0] = 1
    incorrect_counts = np.sum(X_mask, axis=1)
    X_mask[np.arange(num_train), y] = -incorrect_counts
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += reg*W
    return loss, dW

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
