import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold
import xgboost as xgb
from sklearn import linear_model

# 代码参考网络博客
# 导入的包中那些没有用到的是调试阶段中显示直方图用的

df_train = pd.read_csv("datalab/happiness_train_complete.csv", encoding="ansi")
df_test = pd.read_csv("datalab/happiness_test_complete.csv", encoding="ansi")

y_train = df_train['happiness']
y_train = y_train.map(lambda x: 3 if x == -8 else x)
# -----把-8的都统一改成3，表示那些不知道自己幸不幸福的人用中位数代替
# -----因为中位数3的效果测试发现更好，所以没有用众数4
# sns.countplot(x='happiness', data=df_train)
# plt.show()
df_train.drop(["happiness"], axis=1, inplace=True)
# 合并在一起方便处理 训练集和测试集,纵向拼接
df_all = pd.concat((df_train, df_test), axis=0)

# *******************
# 【第一部分：数据处理】
# *******************


# 缺失超过60%的属性删了
df_all.drop(['edu_other', 'invest_other', 'property_other', 'join_party',
             's_work_type', 's_work_status', 'work_status',
             'work_yr', 'work_manage', 'work_type', ], axis=1, inplace=True)
# 全部缺失值填充为0 因为这些缺失值是因为没有结婚而导致的
df_all['s_political'].fillna(0, inplace=True)
df_all['s_hukou'].fillna(0, inplace=True)
df_all['s_income'].fillna(0, inplace=True)
df_all['s_birth'].fillna(0, inplace=True)
df_all['s_edu'].fillna(0, inplace=True)
df_all['s_work_exper'].fillna(0, inplace=True)
# 全部缺失值填充为0 由于没有受过教育造成的
df_all['edu_status'].fillna(0, inplace=True)
df_all['edu_yr'].fillna(0, inplace=True)
# 社交情况 缺失值是由于社交不频繁造成的 全部缺失值填充为7
df_all['social_friend'].fillna(7, inplace=True)
df_all['social_neighbor'].fillna(7, inplace=True)
# 孩子情况 全部填充为0,因为没有孩子
df_all['minor_child'].fillna(0, inplace=True)
# 户口情况 缺失值是由于没有户口造成的 全部缺失值填充为4
df_all['hukou_loc'].fillna(4, inplace=True)
# 婚姻情况填充 因为源数据是2015年的
df_all['marital_now'].fillna(2015, inplace=True)
df_all['marital_1st'].fillna(2015, inplace=True)
# 家庭收入 训练集中的family_income只有一条空记录，用平均值填充
df_all['family_income'].fillna(df_all['family_income'].mean(), inplace=True)

# *******************
# 【第二部分：特征处理】
# *******************


# 宗教不均匀 保留没有删了好，所以删了
df_all.drop(['religion', 'religion_freq'], axis=1, inplace=True)
# 房产不均匀，只保留property_1和property_2效果发现最好
df_all.drop(['property_0', 'property_3', 'property_4', 'property_5',
             'property_6', 'property_7', 'property_8'], axis=1, inplace=True)
# 商业保险、投资、父母政治面貌，网友说删了效果好
df_all.drop(['insur_1', 'insur_2', 'insur_3', 'insur_4'], axis=1, inplace=True)
df_all.drop(['invest_0', 'invest_1', 'invest_2', 'invest_3', 'invest_4',
             'invest_5'], axis=1, inplace=True)
df_all.drop(['f_political', 'm_political'], axis=1, inplace=True)

# *******************
# 【第三部分：构造新特征】
# *******************

df_all['class'] = df_all['class'].map(lambda x: 5 if x == -8 else x)
# sns.countplot(x='class_10_after', data=df_all)
# plt.show()
df_all['edu'] = df_all['edu'].map(lambda x: 0 if x == -8 else x)


# edu分级函数


def edu_split(x):
    if x in [1, 2, 14]:
        return 0
    elif x in [3]:
        return 1
    elif x in [4]:
        return 2
    elif x in [5, 7, 8]:
        return 3
    elif x in [6]:
        return 4
    elif x in [9, 10]:
        return 5
    elif x in [11, 12]:
        return 6
    elif x in [13]:
        return 7


# 意思是edu和s_edu分级后都删了，教育和其配偶教育
df_all["edu"] = df_all["edu"].map(edu_split)
df_all.drop(['edu'], axis=1, inplace=True)
df_all["s_edu"] = df_all["s_edu"].map(edu_split)
# sns.countplot(x='s_edu', data=df_all)
# plt.show()
df_all.drop(['s_edu'], axis=1, inplace=True)

df_all['survey_time'] = pd.to_datetime(df_all['survey_time'], format='%Y-%m-%d %H:%M:%S')
df_all["hour"] = df_all["survey_time"].dt.hour


# 时间分段函数，分段一天的时间


def hour_cut(x):
    if 0 <= x < 6:
        return 0
    elif 6 <= x < 8:
        return 1
    elif 8 <= x < 12:
        return 2
    elif 12 <= x < 14:
        return 3
    elif 14 <= x < 18:
        return 4
    elif 18 <= x < 21:
        return 5
    elif 21 <= x < 24:
        return 6


df_all["hour_cut"] = df_all["hour"].map(hour_cut)
# sns.countplot(x='hour_cut', data=df_all)
# plt.show()
# 构造了 hour 和hour_cut 显然 hour应该被删除
df_all.drop(['hour'], axis=1, inplace=True)


# for i in range(0, 107):
#     print(df_all.columns[i])

# 年代分段函数
def birth_split(x):
    if x < 1920:
        return 0
    if 1920 <= x <= 1930:
        return 1
    elif 1930 < x <= 1940:
        return 2
    elif 1940 < x <= 1950:
        return 3
    elif 1950 < x <= 1960:
        return 4
    elif 1960 < x <= 1970:
        return 5
    elif 1970 < x <= 1980:
        return 6
    elif 1980 < x <= 1990:
        return 7
    elif 1990 < x <= 2000:
        return 8


df_all["birth_s"] = df_all["birth"].map(birth_split)


# sns.countplot(x='birth_s', data=df_all)
# plt.show()


# 年龄划分函数


def age_class(x):
    if x < 0:
        return 0
    elif 0 < x <= 16:
        return 1
    elif 16 < x <= 32:
        return 2
    elif 32 < x <= 48:
        return 3
    elif 48 < x <= 64:
        return 4
    elif 64 < x <= 80:
        return 5
    elif 80 < x <= 96:
        return 6
    else:
        return 7


# 转换时间格式,求出每个人的年龄 分完级去掉age
df_all['age'] = pd.to_datetime(df_all['survey_time']).dt.year - df_all['birth']
df_all["age_class"] = df_all['age'].map(age_class)
# sns.countplot(x='age_class', data=df_all)
# plt.show()
df_all.drop(['age'], axis=1, inplace=True)


# 肥胖划分函数，据说这个很影响幸福感


def get_fat(x):
    if x < 0:
        return 1
    elif 0 < x < 18.5:
        return 1
    elif 18.5 <= x <= 23.9:
        return 2
    elif 24 <= x <= 26.9:
        return 3
    elif 26.9 < x < 29.9:
        return 4
    else:
        return 5


# 借用网友的增加BMI 体重/身高的平方（国际单位kg/㎡） 这边确实厘米和斤
height = df_all["height_cm"] / 100
kg = df_all["weight_jin"] / 2
bmi = kg / pow(height, 2)
df_all["bmi"] = bmi
df_all["fat"] = df_all["bmi"].map(get_fat)
# sns.countplot(x='fat', data=df_all)
# plt.show()
df_all.drop(['bmi'], axis=1, inplace=True)
df_all.drop(['weight_jin'], axis=1, inplace=True)


# 收入分级函数


def get_income_class(x):
    if x <= 0:
        return 0
    if 0 < x < 2800:
        return 1
    elif 2800 <= x < 10000:
        return 2
    elif 10000 <= x < 27000:
        return 3
    elif 27000 <= x < 100000:
        return 4
    else:
        return 5


df_all["income"] = df_all["income"].map(lambda x: 0 if x < 0 else x)
df_all["income_class"] = df_all["income"].map(get_income_class)
# sns.countplot(x='income_class', data=df_all)
# plt.show()
df_all.drop(['income'], axis=1, inplace=True)


# 对住房面积进行划分


def floor_area_split(x):
    if x <= 0:
        return 0
    if 0 < x < 15:
        return 1
    elif 15 <= x < 30:
        return 2
    elif 30 <= x < 50:
        return 3
    elif 50 <= x < 80:
        return 4
    elif 80 <= x < 96:
        return 5
    elif 96 <= x < 110:
        return 6
    elif 110 <= x < 130:
        return 7
    elif 130 <= x < 200:
        return 8
    elif x >= 200:
        return 9


df_all["floor_area_s"] = df_all["floor_area"].map(floor_area_split)


# sns.countplot(x='floor_area_s', data=df_all)
# plt.show()

# 对省份进行分级 因为发现幸福感和省份关系还比较大


def province_split(x):
    if x in [6, 1, 12, 28, 13]:
        return 0
    elif x in [24, 29, 26, 2, 8, 22, 21]:
        return 1
    elif x in [16, 31]:
        return 2
    elif x in [9, 7, 15, 11, 18, 30, 27, 19]:
        return 3
    elif x in [23, 5, 10, 4, 3, 17, 4]:
        return 4


df_all["province_s"] = df_all["province"].map(province_split)
# sns.countplot(x='province_s', data=df_all)
# plt.show()
# 既然已对省份划分 那么省去省份这个属性
df_all.drop(['province'], axis=1, inplace=True)
# 人均住房面积
df_all['aver_area'] = df_all['floor_area'] / df_all['family_m']
# 删除住房面积,人居住房面积
df_all.drop(['floor_area'], axis=1, inplace=True)
df_all.drop(['id'], axis=1, inplace=True)
df_all.drop(['survey_time', 'birth'], axis=1, inplace=True)


def mar_yr_class(x):
    if x <= 0:
        return 0
    elif 0 < x <= 7:
        return 1
    elif 7 < x <= 14:
        return 2
    elif 14 < x < 21:
        return 3
    elif 21 < x <= 28:
        return 4
    elif 28 < x <= 35:
        return 5
    elif 35 < x <= 42:
        return 6
    elif 42 < x <= 49:
        return 7
    elif 49 < x < 56:
        return 8


# 删除其他时间数据
df_all['mar_yr'] = 2015 - df_all['marital_now']
df_all.drop(['s_birth', 'f_birth', 'm_birth'], axis=1, inplace=True)
df_all.drop(['marital_now'], axis=1, inplace=True)
# 网友建议处理一下“公平”效果好
df_all['equity'] = df_all['equity'].map(lambda x: 3 if x == -8 else x)
# sns.countplot(x='equity', data=df_all)
# plt.show()


# 使用z-score方法 进行数据规范化
numeric_cols = ['height_cm', 's_income', 'house', 'family_income', 'family_m',
                'son', 'daughter', 'minor_child', 'inc_exp', 'public_service_1',
                'public_service_2', 'public_service_3', 'public_service_4',
                'public_service_5', 'public_service_6', 'public_service_7',
                'public_service_8', 'public_service_9', 'aver_area', 'mar_yr']
numeric_cols_means = df_all.loc[:, numeric_cols].mean()
numeric_cols_std = df_all.loc[:, numeric_cols].std()
df_numeric = (df_all.loc[:, numeric_cols] - numeric_cols_means) / numeric_cols_std
df_numeric.iloc[:, 1].hist()
# plt.show()
# 对object类型进行规范化  调用get_dummies函数
df_object = df_all.drop(numeric_cols, axis=1)
df_object = df_object.astype(str)
for cols in list(df_object.iloc[:, 1:].columns):
    df_object = pd.get_dummies(df_object.iloc[:, 1:], prefix=cols)

data = pd.concat((df_object, df_numeric), axis=1)

train_data = data.iloc[:8000, :]
test_data = data.iloc[8000:, :]
X_train = train_data.values
X_test = test_data.values
# print(len(X_train))
# print(len(y_train))
# print(len(X_test))

# *******************
# 【第四部分：算法模型选择与融合】
# *******************
# PS：关于模型选择，lightgbm(lgb),xgboost(xgb),catboost(ctb)
# 看一篇博文上进行了融合探究能提升分数
# 然后lgb和xgb融合的效果最好，按照说明验证了下，确实这两个模型组合分数高一点

# 超参数太难调了，哇~
# 实在调不动了，借用网友给的可能最优的参数
# 首先，lgb
param = {'boosting_type': 'gbdt',
         'num_leaves': 20,
         'min_data_in_leaf': 19,
         'objective': 'regression',
         'max_depth': 9,
         'learning_rate': 0.01,
         "min_child_samples": 30,

         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.75,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.15,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros(len(X_test))
# 网上的训练过程，不是特别懂
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])

    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

# print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train)))


# 然后xbg
# 测试用的评价函数myFeval

def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label, preds)
    return 'myFeval', score


xgb_params = {"booster": 'gbtree', 'eta': 0.01, 'max_depth': 5, 'subsample': 0.7,
              'colsample_bytree': 0.6, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8}
folds = KFold(n_splits=2, shuffle=True, random_state=2018)
oof_xgb = np.zeros(8000)
predictions_xgb = np.zeros(2968)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    # print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params, feval=myFeval)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

# print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))

# 【开始融合模型】
# 借用网友的融合代码
# vstack函数作用是垂直堆叠？暂时懂了
train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()
folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2018)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])
for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
    # print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]

    clf_3 = linear_model.BayesianRidge()
    # clf_3 = linear_model.Ridge()
    # clf_3 = linear_model.LinearRegression()
    # clf_3 = linear_model.LassoCV()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10

print("CV score: {:<8.8f}".format(mean_squared_error(oof_stack, y_train)))
test_sub = pd.read_csv("datalab/happiness_submit.csv", encoding='ansi')
result = list(predictions)
test_sub["happiness"] = result
test_sub.to_csv("happiness_submit_2.csv", index=False)
