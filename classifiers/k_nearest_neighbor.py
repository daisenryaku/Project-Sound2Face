import numpy as np

class KNearestNeighbor(object):
  """
  基于K-nearest neighbor算法的监督学习分类器，使用L2 距离
  """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    训练分类器

    Inputs:
    - X：一个[num_train, D]的numpy数组; num_train:训练样本个数，D：样本元素维度
    - y: 一个[num_train,]的numpy数组; y[i]是X[i]的标记
    """
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1):
    """
    对测试集产生预测输出

    Inputs:
    - X：一个[num_test, D]的numpy数组; num_train:测试样本个数，D：样本元素维度
    - k: 近邻个数，即knn算法中超参数k

    Returns:
    - y_pred: 一个[num_test,]的numpy数组; 包含对于测试集的预测输出;y[i]是X[i]的预测标记输出
    """
    dists = self.compute_distances(X)
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      closest_y = []
      closest_y = self.y_train[ np.argsort(dists[i]) ][:k]
      y_pred[i] = np.bincount(closest_y).argmax()
    return y_pred

  def compute_distances(self, X):
    """
    计算X中任意向量与self.X_train中任意向量间的距离
    Inputs:
    - X: 一个[num_test, D]的numpy数组，包含测试集数据

    Returns:
    - dists: 一个[num_test, num_train]的numpy数组; dists[i, j]是测试集中i，训练集中j两个向量间的欧几里得距离
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    xy = -2 * np.dot(X,self.X_train.T)
    x2 = np.sum(X**2,axis=1).reshape(num_test,1)
    y2 = np.sum(self.X_train**2,axis=1).reshape(num_train,1)
    dists = np.sqrt( x2 + xy + y2.T )
    return dists
