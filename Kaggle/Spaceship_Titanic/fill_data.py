import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

data = pd.concat([df_train, df_test]).reset_index(drop=True)
data['first_name'] = data['Name'].apply(lambda x: x.split()[0] if isinstance(x, str) else np.nan)
data['last_name'] = data['Name'].apply(lambda x: x.split()[1] if isinstance(x, str) else np.nan)
data['group'] = data['PassengerId'].apply(lambda x: x.split('_')[0])
data['group_num'] = data['PassengerId'].apply(lambda x: x.split('_')[1])

# print(data)
# data.to_csv('data.csv', index=False)

for col in data.columns.drop(['Transported']):
    print(col)
    print("null: %d, all: %d, %.3f%%" %
          (data[col].isna().sum(), len(data[col]), 100 * data[col].isna().sum() / len(data[col])))
    print()

# tmp = df_train['CryoSleep']
# df_train['CryoSleep'].fillna()

def fill_location(data, col):
    new_location = data.groupby(by=['last_name', 'group']).apply(lambda x: x[col].mode().values[0]
                                if len(x[col].mode().values) != 0 else np.nan).rename('New').reset_index()

    data = data.merge(new_location, on=['last_name', 'group'], how='outer')

    data.loc[data[col].isna(), col] = data[data[col].isna()]['New']
    data.drop(['New'], axis=1, inplace=True)

    new_location = data.groupby(by=['group']).apply(lambda x: x[col].mode().values[0]
                                if len(x[col].mode().values) != 0 else np.nan).rename('New').reset_index()
    data = data.merge(new_location, on=['group'], how='outer')
    data.loc[data[col].isna(), col] = data[data[col].isna()]['New']
    data.drop(['New'], axis=1, inplace=True)

    return data

data_train = data[~data['Transported'].isna()]
data_test = data[data['Transported'].isna()]

#   fill HomePlanet
# print(data['HomePlanet'].isna().sum())
data_train = fill_location(data_train, 'HomePlanet')
data_test = fill_location(data_test, 'HomePlanet')
# print(data['HomePlanet'].isna().sum())
data_train['HomePlanet'].fillna(data_train['HomePlanet'].mode().values[0], inplace=True)
data_test['HomePlanet'].fillna(data_test['HomePlanet'].mode().values[0], inplace=True)

#   fill Destination
# print(data['Destination'].isna().sum())
data_train = fill_location(data_train, 'Destination')
data_test = fill_location(data_test, 'Destination')
# print(data['Destination'].isna().sum())
# data['Destination'].fillna(data['Destination'].mode(), inplace=True)#   doesnt work!???
data_train['Destination'].fillna(data_train['Destination'].mode().values[0], inplace=True)
data_test['Destination'].fillna(data_test['Destination'].mode().values[0], inplace=True)
# data['Destination'] = data['Destination'].fillna(data['Destination'].mode().values[0])
# data.loc[data['Destination'].isna(), 'Destination'] = data['Destination'].mode()#   same!!!????
#   fill pay
data_train.loc[(data_train['CryoSleep'] == True) | (data_train['Age'] < 13), 'RoomService'] = 0.
data_train.loc[(data_train['CryoSleep'] == True) | (data_train['Age'] < 13), 'FoodCourt'] = 0.
data_train.loc[(data_train['CryoSleep'] == True) | (data_train['Age'] < 13), 'ShoppingMall'] = 0.
data_train.loc[(data_train['CryoSleep'] == True) | (data_train['Age'] < 13), 'Spa'] = 0.
data_train.loc[(data_train['CryoSleep'] == True) | (data_train['Age'] < 13), 'VRDeck'] = 0.

data_test.loc[(data_test['CryoSleep'] == True) | (data_test['Age'] < 13), 'RoomService'] = 0.
data_test.loc[(data_test['CryoSleep'] == True) | (data_test['Age'] < 13), 'FoodCourt'] = 0.
data_test.loc[(data_test['CryoSleep'] == True) | (data_test['Age'] < 13), 'ShoppingMall'] = 0.
data_test.loc[(data_test['CryoSleep'] == True) | (data_test['Age'] < 13), 'Spa'] = 0.
data_test.loc[(data_test['CryoSleep'] == True) | (data_test['Age'] < 13), 'VRDeck'] = 0.

data_train['RoomService'].fillna(data_train['RoomService'].median(), inplace=True)
data_train['FoodCourt'].fillna(data_train['FoodCourt'].median(), inplace=True)
data_train['ShoppingMall'].fillna(data_train['ShoppingMall'].median(), inplace=True)
data_train['Spa'].fillna(data_train['Spa'].median(), inplace=True)
data_train['VRDeck'].fillna(data_train['VRDeck'].median(), inplace=True)

data_test['RoomService'].fillna(data_test['RoomService'].median(), inplace=True)
data_test['FoodCourt'].fillna(data_test['FoodCourt'].median(), inplace=True)
data_test['ShoppingMall'].fillna(data_test['ShoppingMall'].median(), inplace=True)
data_test['Spa'].fillna(data_test['Spa'].median(), inplace=True)
data_test['VRDeck'].fillna(data_test['VRDeck'].median(), inplace=True)

data_train['total_pay'] = data_train['RoomService'] + data_train['FoodCourt'] + data_train['ShoppingMall'] + data_train['Spa'] + data_train['VRDeck']
data_test['total_pay'] = data_test['RoomService'] + data_test['FoodCourt'] + data_test['ShoppingMall'] + data_test['Spa'] + data_test['VRDeck']

#   fill CryoSleep
# print(data[data['CryoSleep'] == True][['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].describe(include='all'))
# print(data[(data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck'] == 0)]['CryoSleep'].describe(include='all'))
# print(data[(data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck'] != 0)]['CryoSleep'].describe(include='all'))
# print(data[(data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck'] == 0)
#         & (data['VIP'] == False)].to_string())
# print(data[(data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck'] != 0)
#         & (data['CryoSleep'] == True)].to_string())
# print(data[(data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck'] == 0)
#         & (data['VIP'] == False) & (data['CryoSleep'] == False)].shape[0] / data[(data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck'] == 0)
#         & (data['VIP'] == False)].shape[0])

#   14% неспящих бедных проигнорировано, включая нан
data_train.loc[(data_train['RoomService'] + data_train['FoodCourt'] + data_train['ShoppingMall'] + data_train['Spa'] + data_train['VRDeck'] == 0), 'CryoSleep'] = True
data_train.loc[(data_train['RoomService'] + data_train['FoodCourt'] + data_train['ShoppingMall'] + data_train['Spa'] + data_train['VRDeck'] != 0), 'CryoSleep'] = False
data_test.loc[(data_test['RoomService'] + data_test['FoodCourt'] + data_test['ShoppingMall'] + data_test['Spa'] + data_test['VRDeck'] == 0), 'CryoSleep'] = True
data_test.loc[(data_test['RoomService'] + data_test['FoodCourt'] + data_test['ShoppingMall'] + data_test['Spa'] + data_test['VRDeck'] != 0), 'CryoSleep'] = False

#   fill VIP
# sns.kdeplot(data=data[data['CryoSleep'] == False], x='total_pay', hue="VIP", common_norm=False)
# plt.show()

forest = RandomForestClassifier(random_state=0, n_jobs=-1)

x_train = data_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CryoSleep']][~data_train['VIP'].isna()]
y_train = data_train['VIP'][~data_train['VIP'].isna()].astype('int')
x_test = data_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CryoSleep']][data_train['VIP'].isna()]
x_test_test = data_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CryoSleep']][data_test['VIP'].isna()]
# print(x_train)
# print(data[data['VIP'] == True].to_string())

forest.fit(x_train, y_train)
print('\ttrain acc: %.5f' % accuracy_score(y_train, forest.predict(x_train)))#  0.99716

data_train.loc[data_train['VIP'].isna(), 'VIP'] = forest.predict(x_test)
data_test.loc[data_test['VIP'].isna(), 'VIP'] = forest.predict(x_test_test)
# print(data.loc[data['VIP'].isna(), 'VIP'])

#   fill Age

col_transform = ColumnTransformer(
    transformers=[
        # ('cat', OneHotEncoder(handle_unknown='ignore'), ['HomePlanet', 'Destination']),
        ('num', StandardScaler(), ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                'CryoSleep', 'VIP'])
    ]
)
rtr = Pipeline(steps=[
    ('columnprep', col_transform),
    ('regression', RandomForestRegressor(random_state=0, n_jobs=-1))
    # ('regression', LinearRegression())
    # ('regression', DecisionTreeRegressor())
])

x_train = data_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                'CryoSleep', 'VIP']][~data_train['Age'].isna()]
y_train = data_train['Age'][~data_train['Age'].isna()].astype('int')
x_test = data_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                'CryoSleep', 'VIP']][data_train['Age'].isna()]
x_test_test = data_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                'CryoSleep', 'VIP']][data_test['Age'].isna()]

rtr.fit(x_train, y_train)

print("R2 score for training set: " + str(rtr.score(x_train, y_train)))
# y_pred = rtr.predict(x_train)
data_train.loc[data_train['Age'].isna(), 'Age'] = rtr.predict(x_test)
data_test.loc[data_test['Age'].isna(), 'Age'] = rtr.predict(x_test_test)

# # plot the actual and predicted values of the training set
# plt.scatter(y_train, y_pred)
# plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], c="red")
# plt.xlabel('Age training set')
# plt.ylabel('Age training set predicted')
# plt.axis('equal')
# plt.xlim(0, 100)
# plt.ylim(0, 100)
# plt.show()
#
# # print(data[data['Age'] == 25].to_string())
#
# print('R^2 train: %.3f' % (r2_score(y_train, y_pred)))
#
# plt.scatter(y_pred, y_pred - y_train, c='black', marker='o', label='Train', s=35, alpha=0.5)
# plt.hlines(y=0, xmin=0, xmax=90, lw=2, color='red')
#
# plt.xlim([0, 90])
# plt.xlabel('Предсказанные значения')
# plt.ylabel('Остатки')
# plt.legend(loc='upper left')
# plt.show()

#   fill Cabin
# print(data['Cabin'].isna().sum())
data_train = fill_location(data_train, 'Cabin')
data_test = fill_location(data_test, 'Cabin')
# print(data[data['Transported'] == np.nan]['Cabin'].isna().sum())
# print(data['Cabin'].isna().sum())
data_train.dropna(subset=['Cabin'], inplace=True)
# data_test.dropna(subset=['Cabin'], inplace=True)

#   decrease outliers (1%)
# quantile_values = data_train[['VRDeck', 'Spa', 'RoomService', 'FoodCourt', 'ShoppingMall', 'total_pay', 'Age']].quantile(0.99)
# for col in ['VRDeck', 'Spa', 'RoomService', 'FoodCourt', 'ShoppingMall', 'total_pay', 'Age']:
#     num_values = data_train[col].values
#     threshold = quantile_values[col]
#     num_values = np.where(num_values > threshold, threshold, num_values)
#     data_train[col] = num_values

for col in data.columns.drop(['Transported']):
    print(col)
    print("null: %d, all: %d, %.3f%%" %
          (data_train[col].isna().sum(), len(data_train[col]), 100 * data_train[col].isna().sum() / len(data_train[col])))
    print("null: %d, all: %d, %.3f%%" %
          (data_test[col].isna().sum(), len(data_test[col]), 100 * data_test[col].isna().sum() / len(data_test[col])))
    print()

data = pd.concat([data_train, data_test]).reset_index(drop=True)
data.to_csv('data_filled.csv', index=False)
