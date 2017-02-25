# 听音识人

## 实例数据
```python
from sklearn import datasets
from classifiers.cross_validation import train_test_split
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)
```

## knn 使用实例
```python
from classifiers.k_nearest_neighbor import KNearestNeighbor
model = KNearestNeighbor()
model.train(X_train, y_train)
model.score(X_test, y_test,k=5)
```

## 线性回归
```python
from classifiers.linear_classifier import LinearClassifier
model = LinearClassifier()
model.train(X_train, y_train, learning_rate=1e-3, num_iters=10000, batch_size=20)
model.score(X_test, y_test)
```
