import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import keras

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # 规范y值shape
    n_vars = 1 if type(data) is list else data.shape[1]
    # 格式转化，没有变形
    df = pd.DataFrame(data)

    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        # 从后向前数

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg

# load dataset

# # 转换时间格式
# def parser(x):
#     return datetime.strptime(x, "%Y %m %d %H")
#
# data = pd.read_csv("raw/weather/PRSA_data_2010.1.1-2014.12.31.csv",
#                    parse_dates=[['year', 'month', 'day', 'hour']],
#                    # 合并时间
#                    index_col=0,
#                    # 使时间(第0行)作为index
#                    date_parser=parser
#                    )
# data.drop('No', axis=1, inplace=True)
# # 删除No列（第 1 column数据）
# data = data.dropna()
# # 将所有含有nan项的row删除
# data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# data.index.name = 'date'
# # 修改column 名称
#
# data.to_csv("data/pollution.csv")

data = pd.read_csv("data/pollution.csv", header=0, index_col=0)
# wnd_dict = {}
# for wnd in data['wnd_dir']:
#     if wnd in wnd_dict:
#         wnd_dict[wnd] += 1
#     else:
#         wnd_dict[wnd] = 0
# for index, wnd in enumerate(sorted(wnd_dict)):
#     data = data.replace(wnd, index)





values = data.values
# integer encode direction 对不连续的数字和文本编号
encoder = LabelEncoder()

values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features 规范化数据（把数规定到0到1之间）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 24, 24)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
#print(reframed.head())
values = reframed.values
#print(len(values[0]))
n_train_hours = 365 * 2 * 24
train = values[:n_train_hours, :]
test = values[:n_train_hours, :]

train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]
print(train_x)
print(train_y)
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_x, train_y, epochs=50, batch_size=72, validation_data=(test_x, test_y), verbose=1, shuffle=False)
# plot history



plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')

plt.legend()
plt.show()
