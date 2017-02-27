# Sound2Face

## 实例数据
```python
from datasets import load_breast_cancer
from classifiers import train_test_split
bc = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(bc.data, bc.target, test_size=0.25)
```

## 项目数据
```python
from datasets import load_features
from classifiers import train_test_split
bc = load_features()
X_train, X_test, y_train, y_test = train_test_split(bc.data, bc.target, test_size=0.25)
```

## 数据标准化
```python
from preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
```

## 组合特征值
```python
from classifiers import combine_feature
X_train = combine_feature(X_train)
X_test = combine_feature(X_test)
```

## Logistic回归
```python
from classifiers import LogisticClassifier
model = LogisticClassifier()
model.train(X_train, y_train, learning_rate=1e-3, num_iters=10000, batch_size=100)
model.score(X_test, y_test)
```

## knn 使用实例
```python
from classifiers import KNearestNeighbor
model = KNearestNeighbor()
model.train(X_train, y_train)
model.score(X_test, y_test,k=5)
```

## 线性回归
```python
from classifiers import LinearClassifier
model = LinearClassifier()
model.train(X_train, y_train, learning_rate=1e-3, num_iters=10000, batch_size=100)
model.score(X_test, y_test)
```
## 随机森林
```python
from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier()
rfr.fit(X_train, y_train)
rfr.score(X_test, y_test)

## 支持向量机
```python
from sklearn.svm import LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
lsvc.score(X_test, y_test)
```
## 保存模型
```python
from externals import saveModel, loadModel
saveModelModel(model)
model = loadModel('model.pkl')
```

## 准确度进行评估
```python
from sklearn.metrics import classification_report
y_predict = model.predict(X_test)
print classification_report(y_test,y_predict)
```

## 特征筛选
```python
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
X_test_fs = fs.traform(X_test)
lsvc.fit(X_train_fs, y_train)
lsvc.score(X_test_fs, y_test)
```

