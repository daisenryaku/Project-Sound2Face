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
```pthon
from classifiers import k_nearest_neighbor
model = k_nearest_neighbor.KNearestNeighbor()
model.train(X_train, y_train)
model.score(X_test, y_test,k=5)
```
