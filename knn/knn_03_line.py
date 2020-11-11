from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import numpy as np

reg = KNeighborsRegressor()
reg2 = KNeighborsRegressor(n_neighbors=2)
X, y = make_regression(n_features=1, n_informative=1, noise=50, random_state=8)
reg2.fit(X, y)
# plt.scatter(X, y, c='orange', edgecolors='k')
z = np.linspace(-3, 3, 200).reshape(-1, 1)
plt.scatter(X, y, c='orange', edgecolors='k')
plt.plot(z, reg2.predict(z), c='k', linewidth=3)
# plt.title('KNN Regressor')
plt.title('KNN Regressor: n_neighbors=2')

print()
print('代码运行结果：')
print('================================')
print('模型正确率：{:.2f}'.format(reg2.score(X, y)))
print('================================')
print()

plt.show()
