from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
lr = LinearRegression().fit(X_train, y_train)

print('训练数据集得分：{:.2f}'.format(lr.score(X_train, y_train)))
print('测试数据集得分：{:.2f}'.format(lr.score(X_test, y_test)))
