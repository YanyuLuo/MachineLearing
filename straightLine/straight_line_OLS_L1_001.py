import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
# 使用岭回归对数据进行拟合
rigde = Ridge(alpha=0.1).fit(X_train, y_train)
# 使用套索回归对数据进行拟合
lasso = Lasso().fit(X_train, y_train)

print('训练数据集得分：{:.2f}'.format(lasso.score(X_train, y_train)))
print('测试数据集得分：{:.2f}'.format(lasso.score(X_test, y_test)))
print('套索回归使用的特征数：{}'.format(np.sum(lasso.coef_ != 0)))
