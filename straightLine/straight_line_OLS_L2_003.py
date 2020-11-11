import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
# 使用岭回归对数据进行拟合
rigde = Ridge(alpha=1).fit(X_train, y_train)
rigde01 = Ridge(alpha=0.1).fit(X_train, y_train)
rigde10 = Ridge(alpha=10).fit(X_train, y_train)
lr = LinearRegression().fit(X, y)
plt.plot(rigde.coef_, 's', label='Ridge alpha=1')
plt.plot(rigde10.coef_, '^', label='Ridge alpha=10')
plt.plot(rigde01.coef_, 'v', label='Ridge alpha=0.1')
plt.plot(lr.coef_, 'o', label='linear regression')
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')
plt.hlines(0, 0, len(lr.coef_))
plt.legend()
plt.show()
