import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=50, n_features=1, n_informative=1, noise=50, random_state=1)
reg = LinearRegression()
reg.fit(X, y)
z = np.linspace(-3, 3, 200).reshape(-1, 1)
plt.scatter(X, y, c='b', s=60)
plt.plot(z, reg.predict(z), c='k')
plt.title('Linear Regression')
print('直线的系数是：{:.2f}'.format(reg.coef_[0]))
print('直线的截距是：{:.2f}'.format(reg.intercept_))
plt.show()
