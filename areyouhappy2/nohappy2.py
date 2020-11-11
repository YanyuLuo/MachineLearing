import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

data_train = pd.read_csv("datalab/happiness_train_complete.csv", encoding='gbk')
data_test = pd.read_csv("datalab/happiness_test_complete.csv", encoding='gbk')

data_train['happiness'].unique()
data_train.loc[:, 'happiness'] = list(map(lambda x: x if x > 0 else np.nan, data_train.loc[:, 'happiness']))
data_train.dropna(subset=['happiness'], axis=0, inplace=True)
data_train.reset_index(inplace=True)
data_trains = data_train.iloc[:, 3:]
data_tests = data_test.iloc[:, 1:]
#  合并两个数据集
data = pd.concat([data_trains, data_tests])
data.reset_index(inplace=True)
data = data.iloc[:, 1:]
#  导出合并的数据集于本地
data.to_csv(r'我是合并的数据集.csv', index=False)

columns = data.columns
for column in columns:
    if data[column].dtype != 'object':
        data[column] = list(map(lambda x: x if x > 0 else np.nan, data[column]))

        # print('%s 的异常值个数为：%d' % (column, len(data.loc[data[column] < 0, :][column])))

pd.DataFrame(data.isnull().sum()).tail(50)
#  将缺失值较大的特征删除
data.drop(['edu_yr', 'edu_other', 'join_party', 'property_0',
           'property_1', 'property_2', 'property_3', 'property_4', 'property_5', 'property_6', 'property_7',
           'property_8', 'property_other', 'work_status', 'work_yr', 'work_type', 'work_manage', 'invest_0',
           'invest_1', 'invest_2', 'invest_3', 'invest_4', 'invest_5', 'invest_6', 'invest_7', 'invest_8',
           'invest_other', 'minor_child', 'daughter', 's_work_status', 's_work_type',
           'f_birth', 'm_birth', 'trust_11', 'trust_12'], axis=1, inplace=True)
# print(data.shape)

bir = []
for value in data['birth']:

    if 1921 <= value < 1931:
        bir.append(1)
    elif 1931 <= value < 1941:
        bir.append(2)
    elif 1941 <= value < 1951:
        bir.append(3)
    elif 1951 <= value < 1961:
        bir.append(4)
    elif 1961 <= value < 1971:
        bir.append(5)
    elif 1971 <= value < 1981:
        bir.append(6)
    elif 1981 <= value <= 1991:
        bir.append(7)
    elif 1991 <= value <= 2001:
        bir.append(8)
data['birth'] = pd.DataFrame(bir)

#  依据特征income创造收入人群称呼
incomes = []
for income in data['income']:
    if 0 <= income < 200000:
        incomes.append(1)  # 困难家庭
    elif 200000 <= income < 350000:
        incomes.append(2)  # 贫困家庭
    elif 350000 <= income < 600000:
        incomes.append(3)  # 贫穷家庭
    elif 600000 <= income < 800000:
        incomes.append(4)  # 小康家庭
    elif 800000 <= income < 2000000:
        incomes.append(5)  # 中产家庭
    elif 2000000 <= income < 5000000:
        incomes.append(6)  # 富裕家庭
    elif 5000000 <= income:
        incomes.append(7)  # 富人家庭

data['income'] = pd.DataFrame(incomes)

#  依据特征family_income创造收入人群称呼
family_incomes = []
for income in data['family_income']:
    if 0 <= income < 200000:
        family_incomes.append(1)  # 困难家庭
    elif 200000 <= income < 350000:
        family_incomes.append(2)  # 贫困家庭
    elif 350000 <= income < 600000:
        family_incomes.append(3)  # 贫穷家庭
    elif 600000 <= income < 800000:
        family_incomes.append(4)  # 小康家庭
    elif 800000 <= income < 2000000:
        family_incomes.append(5)  # 中产家庭
    elif 2000000 <= income < 5000000:
        family_incomes.append(6)  # 富裕家庭
    elif 5000000 <= income:
        family_incomes.append(7)  # 富人家庭

data['family_income'] = pd.DataFrame(family_incomes)

#  依据特征s_income创造收入人群称呼
s_incomes = []
for income in data['s_income']:
    if 0 <= income < 200000:
        s_incomes.append(1)  # 困难家庭
    elif 200000 <= income < 350000:
        s_incomes.append(2)  # 贫困家庭
    elif 350000 <= income < 600000:
        s_incomes.append(3)  # 贫穷家庭
    elif 600000 <= income < 800000:
        s_incomes.append(4)  # 小康家庭
    elif 800000 <= income < 2000000:
        s_incomes.append(5)  # 中产家庭
    elif 2000000 <= income < 5000000:
        s_incomes.append(6)  # 富裕家庭
    elif 5000000 <= income:
        s_incomes.append(7)  # 富人家庭

data['s_income'] = pd.DataFrame(s_incomes)

#  依据特征inc_exp创造消费人群称呼
inc_exps = []
for income in data['inc_exp']:
    if 0 <= income < 200000:
        inc_exps.append(1)  # 困难家庭
    elif 200000 <= income < 350000:
        inc_exps.append(2)  # 贫困家庭
    elif 350000 <= income < 600000:
        inc_exps.append(3)  # 贫穷家庭
    elif 600000 <= income < 800000:
        inc_exps.append(4)  # 小康家庭
    elif 800000 <= income < 2000000:
        inc_exps.append(5)  # 中产家庭
    elif 2000000 <= income < 5000000:
        inc_exps.append(6)  # 富裕家庭
    elif 5000000 <= income:
        inc_exps.append(7)  # 富人家庭

data['inc_exp'] = pd.DataFrame(inc_exps)

data.drop(['survey_time', 'survey_type', 'province', 'city', 'county',
           'marital_1st', 's_birth', 'marital_now'], axis=1, inplace=True)
# print(data.shape)
pd.DataFrame(data.isnull().sum())
a = pd.DataFrame(data.isnull().sum() != 0)
a = a[a.values == True]
# print(a.index)
#  把专门的缺失值的特征跳出来，形成一个新的表格。
datas_pre = pd.DataFrame()
for col in a.index:
    datas_pre[col] = data[col]
# print(datas_pre.shape)

np.argsort(datas_pre.isnull().sum(axis=0)).values

datas_pre['inc_ability'] = pd.to_numeric(datas_pre['inc_ability'])

y_full = data_train['happiness']
print(y_full)
data_pre_reg = datas_pre.copy()
sort_index = np.argsort(data_pre_reg.isnull().sum(axis=0)).values

data_pre_reg.columns = [x for x in range(len(data_pre_reg.columns))]

# for i in sort_index:
#     df = data_pre_reg
#     #  构建新标签
#     fillc = df.iloc[:, i]
#     #  构建新特征矩阵
#     df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)
#     #  对于新的特征矩阵中，用0进行填充
#     imp_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
#     df_0 = pd.DataFrame(imp_0.fit_transform(df))
#     #  挑选出不缺失的标签
#     Ytrain = fillc[fillc.notnull()]
#     #  需要Ytest的index啊
#     Ytest = fillc[fillc.isnull()]
#     Xtrain = df_0.iloc[Ytrain.index, :]
#     Xtest = df_0.iloc[Ytest.index, :]
#     #  建立随机森林回归模型
#     rfc = RandomForestRegressor(n_estimators=100)
#     rfc = rfc.fit(Xtrain, Ytrain)
#     Ypredict = rfc.predict(Xtest)
#
#     data_pre_reg.loc[data_pre_reg.iloc[:, i].isnull(), i] = Ypredict
#
# #  将data_pre_reg归进去data里。
# data_pre_reg.columns = a.index
#
# for column in data_pre_reg.columns:
#     data[column] = data_pre_reg[column]
#
# data.to_csv(r'完整版啊.csv', index=False)
#
# #  剔除相关度较低的特征
# data.drop(['religion_freq', 'work_exper', 'income', 'inc_exp', 's_hukou'], axis=1, inplace=True)
#
# X_train = data.iloc[:7988, :]
# X_test = data.iloc[7988:, :]
# X_test.to_csv(r'test_data_complete.csv', index=False)
# train_data_complete = pd.DataFrame(X_train, y_full)
# train_data_complete.to_csv(r'train_data_complete.csv', index=False)
