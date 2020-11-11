import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor

import xgboost as xgb

data = pd.read_csv("完整版啊.csv", encoding='gbk')
data.drop(['religion_freq', 'work_exper', 'income', 'inc_exp', 's_hukou'], axis=1, inplace=True)
X_train = data.iloc[:7988, :]
X_test = data.iloc[7988:, :]

data_train = pd.read_csv("datalab/happiness_train_complete.csv", encoding='gbk')
data_train['happiness'].unique()
data_train.loc[:, 'happiness'] = list(map(lambda x: x if x > 0 else np.nan, data_train.loc[:, 'happiness']))
data_train.dropna(subset=['happiness'], axis=0, inplace=True)
data_train.reset_index(inplace=True)
y_full = data_train['happiness']

# print(X_train)
# print('--------------------------')
# print(y_full)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, y_full, test_size=0.3, random_state=1227)
xgboo = XGBRegressor().fit(Xtrain, ytrain)
train_d = xgb.DMatrix(Xtrain, ytrain)

# #  默认随机挑选参数
# xgb1 = XGBRegressor(max_depth=3,
#                     learning_rate=0.1,
#                     n_estimators=5000,
#                     silent=False,
#                     booster='gbtree',
#                     objective='reg:squarederror',
#                     n_jobs=4,
#                     gamma=0,
#                     min_child_weight=1,
#                     subsample=0.8,
#                     colsample_bytree=0.8,
#                     seed=7)
#
# # print(xgb1.get_params())
# #  利用cv函数选择最佳的树的数量。
# print(xgb.cv(xgb1.get_xgb_params(), train_d, xgb1.get_params()['n_estimators'], nfold=5,
#              metrics='rmse', early_stopping_rounds=70)
#       )

#  更新树的数量的参数并计算mse
# xgb1 = XGBRegressor(max_depth=3,
#                     learning_rate=0.1,
#                     n_estimators=169,
#                     silent=False,
#                     objective='reg:squarederror',
#                     booster='gbtree',
#                     n_jobs=4,
#                     gamma=0,
#                     min_child_weight=1,
#                     subsample=0.8,
#                     colsample_bytree=0.8,
#                     seed=7)
# #  然后训练模型、测试集预测、获得mse得分
# xgb1_best1 = xgb1.fit(Xtrain, ytrain)
#
# predict = xgb1_best1.predict(Xtest)
# print('The best mse:', mean_squared_error(ytest, predict))
# print('The best r2:', r2_score(ytest, predict))
# print('-----------')
#
# param = {
#     'max_depth': [1, 2, 3, 4, 5, 6],
#     'min_child_weight': [1, 2, 3, 4, 5, 6]
# }
#
# grid = GridSearchCV(xgb1, param_grid=param, cv=5)
# grid.fit(Xtrain, ytrain)
# print('The Best Params:', grid.best_params_)
# print('The Best Score:', grid.best_score_)

#  gamma参数调优

# xgb1 = XGBRegressor(max_depth=2,
#                     learning_rate=0.01,
#                     n_estimators=2379,
#                     silent=False,
#                     objective='reg:squarederror',
#                     booster='gbtree',
#                     n_jobs=4,
#                     gamma=1.8,
#                     min_child_weight=2,
#                     subsample=0.8,
#                     colsample_bytree=0.3,
#                     reg_lambda=1.54,
#                     seed=7)
#
# print(xgb.cv(xgb1.get_xgb_params(), train_d, xgb1.get_params()['n_estimators'], nfold=5,
#              metrics='rmse', early_stopping_rounds=60)
#       )

xgb1 = XGBRegressor(max_depth=2,
                    learning_rate=0.01,
                    n_estimators=2379,
                    silent=False,
                    objective='reg:squarederror',
                    booster='gbtree',
                    n_jobs=4,
                    gamma=1.8,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.3,
                    reg_lambda=1.54,
                    seed=7)

#  然后训练模型、测试集预测、获得r2得分
xgb1_best2 = xgb1.fit(Xtrain, ytrain)

predicts = xgb1_best2.predict(Xtest)

print('最优模型的mse:', mean_squared_error(ytest, predicts))
print('最优模型的r2:', r2_score(ytest, predicts))

model = xgb1.fit(Xtrain, ytrain)
model.save_model(r'The Best Model')
pre = model.predict(X_test)
pd.DataFrame(pre).to_csv(r'resultss.csv',index=False)
