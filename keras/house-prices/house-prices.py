# 导入需要的模块

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 用来绘图的，封装了matplot
# 要注意的是一旦导入了seaborn，
# matplotlib的默认作图风格就会被覆盖成seaborn的格式
from matplotlib import pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 为了在jupyter notebook里作图，需要用到这个命令
data_train = pd.read_csv("./raw/train.csv")
print(data_train['SalePrice'].describe())
#skewness and kurtosis  偏度 、 峰度
print("Skewness: %f" % data_train['SalePrice'].skew())
print("Kurtosis: %f" % data_train['SalePrice'].kurt())
#plt.plot(data_train['SalePrice']) # 线性
#plt.hist(data_train['SalePrice'], 50)# 柱状


# 单个特征查看是否主要
# var = "CentralAir"
#
# data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)
# # plt.xlabel(var)
# # plt.ylabel(y)
# # plt.boxplot(data)
# # plt.axis(ymin=0, ymax=80000)
# plt.show()


# 显示相关性矩阵
# corrmat = data_train.corr()
# f, ax = plt.subplots(figsize=(20, 9))
# sns.heatmap(corrmat, vmax=0.8, square=True)
# plt.show()

# 显示前十个与SalePrice 相关性强的 因子
# corrmat = data_train.corr()
# k  = 10 # 关系矩阵中将显示10个特征
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(data_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, \
#                  square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

# 各个特征关系点图
# sns.set()
# cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
# sns.pairplot(data_train[cols], size = 2.5)
# plt.show()


#