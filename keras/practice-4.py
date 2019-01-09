import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from os import path
from PIL import Image
# LabelEncoder 使用谨慎，建议不要使用—_—
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, minmax_scale
# import plotly modules
import plotly.offline as py
from sklearn.model_selection import train_test_split


py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd
def isNum(value):
    try:
        value + 1
    except TypeError:
        return False
    else:
        return True
train = pd.read_csv("./raw/HousePrices/train.csv")
test = pd.read_csv("./raw/HousePrices/test.csv")
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

print(train.describe())# 数据分析
print(train.isnull().sum())
print(train.columns)
# train = pd.get_dummies(train) #数据one-hot 模式会添加数据到columns
# test = pd.get_dummies(test)
y = train['SalePrice']
train = train.drop(['SalePrice'], axis=1)

print(train.values)


print(train.head(5))
print(train.isnull().sum())
# mean = train.mean().astype(np.int32)# 求平均值
# train.fillna(mean, inplace=True)# 使用平均值填充nun
#
# mean = test.mean().astype(np.int32)
# test.fillna(mean , inplace = True)
id = test['Id']
test.drop(['Id'], axis=1, inplace=True)
train.drop(['Id'], axis=1, inplace=True)
encoders = {}
train['Alley'].fillna("N", inplace=True)
train['MasVnrType'].fillna("N", inplace=True)
train['BsmtQual'].fillna("N", inplace=True)
train['BsmtCond'].fillna("N", inplace=True)
train['BsmtExposure'].fillna("N", inplace=True)
train['BsmtFinType1'].fillna("N", inplace=True)
train['BsmtFinType2'].fillna("N", inplace=True)
train['Electrical'].fillna("N", inplace=True)
train['FireplaceQu'].fillna("N", inplace=True)
train['GarageType'].fillna("N", inplace=True)
train['GarageFinish'].fillna("N", inplace=True)
train['GarageQual'].fillna("N", inplace=True)
train['GarageCond'].fillna("N", inplace=True)
train['PoolQC'].fillna("N", inplace=True)
train['Fence'].fillna("N", inplace=True)
train['MiscFeature'].fillna("N", inplace=True)

test['Alley'].fillna("N", inplace=True)
test['MasVnrType'].fillna("N", inplace=True)
test['BsmtQual'].fillna("N", inplace=True)
test['BsmtCond'].fillna("N", inplace=True)
test['BsmtExposure'].fillna("N", inplace=True)
test['BsmtFinType1'].fillna("N", inplace=True)
test['BsmtFinType2'].fillna("N", inplace=True)
test['Electrical'].fillna("N", inplace=True)
test['FireplaceQu'].fillna("N", inplace=True)
test['GarageType'].fillna("N", inplace=True)
test['GarageFinish'].fillna("N", inplace=True)
test['GarageQual'].fillna("N", inplace=True)
test['GarageCond'].fillna("N", inplace=True)
test['PoolQC'].fillna("N", inplace=True)
test['Fence'].fillna("N", inplace=True)
test['MiscFeature'].fillna("N", inplace=True)
test.fillna(0, inplace=True)
for colums_name in train.columns:

    encoder = LabelEncoder()
    if not isNum(train[colums_name][0]):
        print(colums_name)
        col = encoder.fit_transform(train[colums_name])
        encoders[colums_name] = encoder
        train[colums_name] = col
    else:
        train[colums_name].fillna(0, inplace=True)
        train[colums_name].astype('int32')
scaler = MinMaxScaler()

train = pd.DataFrame(scaler.fit_transform(train.values), columns=train.columns)



x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.05, random_state=42)

print(x_train.shape)
print(y_train.shape)



def train():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(250, input_shape=(x_train.shape[1],)))
    model.add(keras.layers.Activation('linear'))
    model.add(keras.layers.Dropout(0.2))#relu
    model.add(keras.layers.Dense(125))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Dense(1, activation="linear"))
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # 优化函数，设定学习率（lr）等参数
    model.compile(loss='mae', optimizer=sgd)
    model.fit(x_train, y_train, epochs=10000, batch_size=32, validation_data=(x_test, y_test), verbose=2, shuffle=False)
    model.save("data/practice-4.h5")

train()

def predicte():
    model = keras.models.load_model("data/practice-4.h5")

    for i in range(len(test)):
        print(i)
        print(test.values[i])
        x = test.loc[i]

        for col_name in encoders:

            n = encoders[col_name].transform([x[col_name]])
            #print(n)
            x[col_name] = n[0]
        print(x.values)
        r = scaler.transform([x.values])

        r = r.reshape((1,79,))
        preds = model.predict(x=r)
        print(preds)


#predicte()
# sub = pd.DataFrame({"Id": id ,"SalePrice": reg.predict(test.values).round(3)})
# sub.to_csv("stackcv_linearsvc.csv", index=False)