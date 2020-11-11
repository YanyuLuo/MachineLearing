import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
# 使用岭回归对数据进行拟合
rigde = Ridge(alpha=1).fit(X_train, y_train)
rigde01 = Ridge(alpha=0.1).fit(X_train, y_train)
rigde10 = Ridge(alpha=10).fit(X_train, y_train)
lasso = Lasso(alpha=1).fit(X_train, y_train)
lasso01 = Lasso(alpha=0.1).fit(X_train, y_train)
lasso00001 = Lasso(alpha=0.0001).fit(X_train, y_train)

plt.plot(lasso.coef_, 's', label='Lasso alpha=1')
plt.plot(lasso01.coef_, '^', label='Lasso alpha=0.1')
plt.plot(lasso00001.coef_, 'v', label='Lasso alpha=0.0001')
plt.plot(rigde01.coef_, 'o', label='Ridge alpha=0.1')
plt.ylim(-1100, 850)
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')
plt.legend(ncol=2, loc=(0, 1.05))
plt.show()
