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
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd



class YMaxMinScaler():
    max_number, min_number = 0, 0
    data = []

    def init_max_min(self, data):
        for index, number in enumerate(data):
            if index == 0:
                self.max_number = number
                self.min_number = number
            if self.max_number < number:
                self.max_number = number
            if self.min_number > number:
                self.min_number = number

    def fit_transform(self, data):
        self.data = data.sort()
        self.init_max_min(data)
        return_data = []
        for number in data:
            return_data.append((number-self.min_number)/(self.max_number-self.min_number))
        return return_data

    def inverse_transform(self, data):
        number = data*(self.max_number-self.min_number)+self.min_number
        index = 0
        for i, n in enumerate(self.data):
            if n > number :
                index = i
                break
        if self.data[index]- number > number - self.data[index-1]:
            number = self.data[index-1]
        else:
            number = self.data[index]
        return number

cups = pd.read_csv("raw/WorldCup/WorldCups.csv")
players = pd.read_csv("raw/WorldCup/WorldCupPlayers.csv")
matches = pd.read_csv("raw/WorldCup/WorldCupMatches.csv")

# DROP NA VALUES
# drop 为 NAN 数据

# players = players.dropna()
players = players.drop_duplicates()
cups = cups.dropna()
cups = cups.drop_duplicates()
# matches = matches.dropna()
matches = matches.drop_duplicates()

# 统一西德和德国称呼
cups = cups.replace("Germany FR", "Germany")
players = players.replace("Germany FR", "Germany")
matches = matches.replace("Germany FR", "Germany")
matches = matches.replace("German DR", "Germany")
# 修改一些乱码字符
# ['RoundID', 'MatchID', 'Team Initials', 'Coach Name', 'Line-up', 'Shirt Number', 'Player Name', 'Position', 'Event']

# ['Year', 'Datetime', 'Stage', 'Stadium', 'City', 'Home Team Name', 'Home Team Goals',
# 'Away Team Goals', 'Away Team Name', 'Win conditions', 'Attendance', 'Half-time Home Goals', 'Half-time Away Goals',
# 'Referee', 'Assistant 1', 'Assistant 2', 'RoundID', 'MatchID', 'Home Team Initials', 'Away Team Initials']

# 删除不要项
del matches['Year']
del matches['Datetime']
del matches['Stage']
del matches['Stadium']
del matches['City']
del matches['Home Team Name']
# Home Team Goals
# Away Team Goals
del matches['Away Team Name']
del matches['Win conditions']
del matches['Attendance']
# Half-time Home Goals
# Half-time Away Goals
del matches['Referee']
del matches['Assistant 1']
del matches['Assistant 2']
del matches['RoundID']
del players['RoundID']
cups['Attendance'] = cups['Attendance'].str.replace(".", "").astype("int64")

team_initials_encoder = LabelEncoder()
team_initials_dictionary = team_initials_encoder.fit_transform(players['Team Initials'])
team_initials_dictionary = dict(zip(players['Team Initials'], team_initials_dictionary))

matchID_scaler = YMaxMinScaler()
print(300186460 in matches['MatchID'].values)

matchID_dictionary = matchID_scaler.fit_transform(matches['MatchID'].values)

matchID_dictionary = dict(zip(matches['MatchID'], matchID_dictionary))
print(matchID_dictionary)


def isNum(value):
    try:
        value + 1
    except TypeError:
        return False
    else:
        return True


def transformation_data_to_digital(data, isToOther, columnNumber):

    encoders = []
    values = data.values
    names = data.columns
    # print(values)
    # values = encoder.fit_transform(values)
    # normalize features 规范化数据（把数规定到0到1之间）

    scaler = MinMaxScaler()  # feature_range=(0, 1)
    for i in range(len(names)):
        # integer encode direction 对不连续的数字和文本编号
        encoder = LabelEncoder()
        if "Team Initials" in names[i]:
            for index, body in enumerate(data.iloc[:, i]):
                data.iloc[index, i] = team_initials_dictionary[data.iloc[index, i]]
            continue
        if "MatchID" in names[i]:
            for index, body in enumerate(data.iloc[:, i]):

                data.iloc[index, i] = matchID_dictionary[data.iloc[index, i]]
            continue
        if isNum(data.iloc[0, i]) and not pd.isnull(data.iloc[0, i]):
            continue
        if isToOther:
            flag = False
            for n in range(len(columnNumber)):
                if columnNumber[n] == i:
                    flag = True
            if flag:
                for j in range(len(data.iloc[:, i])):
                    if pd.isnull(data.iloc[j, i]):
                        data.iloc[j, i] = " "
            data.iloc[:, i] = encoder.fit_transform(data.iloc[:, i])
            data.iloc[:, i] = data.iloc[:, i].astype('float64')
            encoders.append(encoder)
        else:
            data.iloc[:, i] = encoder.fit_transform(data.iloc[:, i])
            data.iloc[:, i] = data.iloc[:, i].astype('float64')
            encoders.append(encoder)

    data.iloc[:, :] = scaler.fit_transform(data.iloc[:, :])
    # ensure all data is float
    # values = values.astype('float32')
    return scaler, encoders


(matches_scaler, matches_encoder) = transformation_data_to_digital(matches, False, None)
(players_scaler, players_encoder) = transformation_data_to_digital(players, True, [6, 7])
print(matches)
print(players)
# (matches_values, matches_scaler, matches_encoder) = transformation_data_to_digital(matches, False, None)
# (players_values, players_scaler, players_encoder) = transformation_data_to_digital(players, True, [7, 8])
# (cups_values, cups_scaler, cups_encoder) = transformation_data_to_digital(cups, False, None)

# matches_values
# scaler_matches_matchID = matches_values[:, 12]
# scaler_players_matchID = players_values[:, 1]



# matches_values[:, 12] = matches['MatchID']
# players_values[:, 1] = players["MatchID"]


# matches_name = matches.columns
# matches = pd.DataFrame(matches_values)
# matches.columns = matches_name
# players_name = players.columns
# players = pd.DataFrame(players_values)
# players.columns = players_name
# cups_name = cups.columns
# cups = pd.DataFrame(cups_values)
# cups.columns = cups_name


def generated_player_by_match(matchID):

    match = matches[matches.MatchID == matchID]

    match_players = players[(players.MatchID == matchID)]

    return match.values, match_players.values

length = 0
for i, matches_and_players in matches.iterrows():
    match, match_players = generated_player_by_match(matches_and_players['MatchID'])

    if len(match_players) > length:
        length = len(match_players)

train_matches = []
train_players = []

for i, matches_and_players in matches.iterrows():
    match, match_players = generated_player_by_match(matches_and_players['MatchID'])
    m_p = np.zeros((length, 8))
    # del match['MatchID']
    # del match_players['MatchID']
    for index, player in enumerate(match_players):
        m_p[index, :] = player
    train_matches.append(match)
    train_players.append(m_p)
train_players = np.array(train_players)
train_matches = np.array(train_matches)
print("train_matches", train_matches.shape)
print("train_player", train_players.shape)
print(players.columns)
# (players_scaler, players_encoder) = transformation_data_to_digital(players, True, [6, 7])
# (cups_scaler, cups_encoder) = transformation_data_to_digital(cups, False, None)
# x[ 46, 9]
# y [15]

train_matches = train_matches.reshape((len(train_matches), 7))
total_matches = len(train_matches)
train_number = 672

train_matches = train_matches.reshape((-1, 7))
train_players = train_players.reshape((-1, 368))

train_x = train_players[: train_number]
train_y = train_matches[: train_number]
test_x = train_players[train_number:]
test_y = train_matches[train_number:]
print(train_players.shape, train_matches.shape)

def train():

    model = keras.models.Sequential()
    # model.add(keras.layers.Masking(mask_value=0., input_shape=414))
    model.add(keras.layers.BatchNormalization(input_shape=(368,), momentum=0.99))
    model.add(keras.layers.Dense(500, input_shape=(368,)))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(500))
    model.add(keras.layers.Activation('tanh'))
    # model.add(keras.layers.Dense(1000))
    # model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(7, activation="relu"))
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # 优化函数，设定学习率（lr）等参数
    model.compile(loss='mae', optimizer=sgd)
    model.fit(train_x, train_y, epochs=10000, batch_size=32, validation_data=(test_x, test_y), verbose=2, shuffle=False)
    model.save("data/practice-2.h5")

# 可以使用cnn模型， 类似图片处理
# 现在主要问题是keras的shape 维度处理问题， 不清楚keras如何降维

def predicte():
    model = keras.models.load_model("data/practice-2.h5")
    x = train_players[0]
    x = x.reshape((1, 368,))
    preds = model.predict(x=x)
    # print("Predicted:", preds)
    preds2 = matches_scaler.inverse_transform(preds)
    print("Predicted 比分:"+str(preds2[0][0])+":"+str(preds2[0][1]))
    print("Predicted 半场比分:"+str(preds2[0][2])+":"+str(preds2[0][3]))
    # print(len(matches_encoder))
predicte()
# train()
