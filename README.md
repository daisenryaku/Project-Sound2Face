# 听音识人

## 实例数据
```python
from sklearn import datasets
iris = datasets.load_iris()
num_train = 125
num_test = 25
X_train = iris.data[:num_train]
y_train = iris.target[:num_train]
X_test = iris.data[-num_test:]
y_test = iris.target[-num_test:]
```

## knn 使用实例
```python
from classifiers import k_nearest_neighbor
model = k_nearest_neighbor.KNearestNeighbor()
model.train(X_train, y_train)
model.score(X_test, y_test,k=5)
```

## 线性回归
```python
from classifiers import linear_classifier
model = linear_classifier.LinearClassifier()
model.train(X_train, y_train, learning_rate=1e-3, num_iters=10000, batch_size=20)
model.score(X_test, y_test)
```
