import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from Spaceship_Titanic.my_lib import *
import missingno as msno


# df_train = pd.read_csv('../data/train.csv')
# df_test = pd.read_csv('../data/test.csv')
# print(msno.matrix(df_train))
# print(df_train.describe(include='all').transpose().to_string())
# print(df_train.shape)
# print(df_train)
# print(df_test.describe().transpose().to_string())

data = pd.read_csv('data_filled.csv')

def circle_and_bar(train, test, feature):
        hp_train = dict(Counter(train[feature].fillna(value='nan')))
        hp_test = dict(Counter(test[feature].fillna(value='nan')))
        print(hp_train)

        labels = list(hp_train.keys())

        fig = plt.figure(feature)
        ax1 = fig.add_subplot(221)
        ax1.title.set_text('Train')
        plt.pie([hp_train[i] for i in labels],
                labels = labels,
                colors = sns.color_palette('pastel')[:len(labels)],
                autopct='%.2f%%')
        ax2 = fig.add_subplot(222)
        ax2.title.set_text('Test')
        plt.pie([hp_test[i] for i in labels],
                labels = labels,
                colors = sns.color_palette('pastel')[:len(labels)],
                autopct='%.2f%%')

        ax3 = fig.add_subplot(223)
        ax=sns.countplot(data=train.fillna(value='nan'), x=feature, hue='Transported')
        for p in ax.patches:
            ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))
        ax.legend(title='Transported', loc='upper right')
        plt.show()


def show_high_corr(data):
    cols = data.columns[~data.columns.isin(['user_id', 'will_stay'])]
    corr_matrix = data[cols].corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any((upper[column] > 0.9) | (upper[column] < -0.9))]

    for element in to_drop:
        column_list = list(data[cols].columns[np.where(
            (data[cols].corrwith(data[cols][element]) > 0.9) |
            (data[cols].corrwith(data[cols][element]) < -0.9))])
        column_list.remove(element)
        for column in column_list:
            print(str(element) + " <-> " + str(column) + ": " + str(data[cols][element].corr(data[cols][column])))

#       analyse categorical features
# circle_and_bar(df_train, df_test, 'HomePlanet')
# circle_and_bar(df_train, df_test, 'CryoSleep')
# circle_and_bar(df_train, df_test, 'Destination')
# circle_and_bar(df_train, df_test, 'VIP')

# sns.kdeplot(data=df_train, x="Age", hue="Transported", common_norm=False)
# plt.xlim(0, df_train['Age'].max())
# plt.grid()
# plt.show()
sns.kdeplot(data=data, x="Age", hue="Transported", common_norm=False)
plt.xlim(0, data['Age'].max())
plt.grid()
plt.show()

def age_category(age):
    if age <= 8:
        return 'child'
    if 8 < age <= 18.5:
        return 'schoolboy'
    if 18.5 < age <= 26:
        return 'student'
    if 26 < age <= 40:
        return 'adult'
    if age > 40:
        return 'senior'
    else:
        return 'no age'

# df_train['Age_category'] = df_train['Age'].apply(lambda x: age_category(x))
# df_test['Age_category'] = df_test['Age'].apply(lambda x: age_category(x))
data['Age_category'] = data['Age'].apply(lambda x: age_category(x))

# pivot_survival_rate(df_train, 'HomePlanet')
# pivot_survival_rate(df_train, 'CryoSleep')
pivot_survival_rate(data[~data['Transported'].isna()], 'CryoSleep')
# pivot_survival_rate(df_train, 'Destination')
# pivot_survival_rate(df_train, 'VIP')
# pivot_survival_rate(df_train, 'Age_category')


# df_train['Cabin_deck'] = df_train['Cabin'].apply(lambda x: x.split('/')[0] if isinstance(x, str) else 'nan')
# df_train['Cabin_num'] = df_train['Cabin'].apply(lambda x: int(x.split('/')[1]) if isinstance(x, str) else None)
# df_train['Cabin_side'] = df_train['Cabin'].apply(lambda x: x.split('/')[2] if isinstance(x, str) else 'nan')
# df_test['Cabin_deck'] = df_test['Cabin'].apply(lambda x: x.split('/')[0] if isinstance(x, str) else 'nan')
# df_test['Cabin_num'] = df_test['Cabin'].apply(lambda x: int(x.split('/')[1]) if isinstance(x, str) else None)
# df_test['Cabin_side'] = df_test['Cabin'].apply(lambda x: x.split('/')[2] if isinstance(x, str) else 'nan')
data['Cabin_deck'] = data['Cabin'].apply(lambda x: x.split('/')[0] if isinstance(x, str) else 'nan')
data['Cabin_num'] = data['Cabin'].apply(lambda x: int(x.split('/')[1]) if isinstance(x, str) else None)
data['Cabin_side'] = data['Cabin'].apply(lambda x: x.split('/')[2] if isinstance(x, str) else 'nan')

# pivot_survival_rate(df_train[df_train['CryoSleep'] == 1], 'Cabin_deck')
pivot_survival_rate(data[(~data['Transported'].isna()) & (data['CryoSleep'] == 1)], 'Cabin_deck')
# pivot_survival_rate(df_train, 'Cabin_num')
# pivot_survival_rate(df_train[df_train['CryoSleep'] == 1], 'Cabin_side')
pivot_survival_rate(data[(~data['Transported'].isna()) & (data['CryoSleep'] == 1)], 'Cabin_side')

# df_train['Cabin_deck_side'] = df_train['Cabin'].apply(lambda x: x.split('/')[0] + x.split('/')[2] if isinstance(x, str) else 'nan')
data['Cabin_deck_side'] = data['Cabin'].apply(lambda x: x.split('/')[0] + x.split('/')[2] if isinstance(x, str) else 'nan')
# df_test['Cabin_deck_side'] = df_test['Cabin'].apply(lambda x: x.split('/')[0] + x.split('/')[2] if isinstance(x, str) else 'nan')

# sns.kdeplot(data=df_train[(df_train['Cabin_deck_side'] == 'GS') & (df_train['CryoSleep'] == 1)], x="Cabin_num", hue="Transported", common_norm=False)
# plt.xlim(0, df_train['Cabin_num'][df_train['Cabin_deck_side'] == 'GS'].max())
# plt.grid()
# plt.show()

# pivot_survival_rate(df_train[df_train['CryoSleep'] == 1], 'Cabin_deck_side')
pivot_survival_rate(data[(~data['Transported'].isna()) & (data['CryoSleep'] == 1)], 'Cabin_deck_side')

# df_train['in_danger_zone'] = False
data['in_danger_zone'] = False
# df_test['in_danger_zone'] = False

# print(sorted(df_train['Cabin_num'][(df_train['Cabin_deck'] == 'A') & (df_train['Cabin_side'] == 'P')]))
# print(sorted(df_train['Cabin_num'][(df_train['Cabin_deck'] == 'A') & (df_train['Cabin_side'] == 'S')]))
# print(df_train)
#
# print(df_train.iloc[:10].to_string())
# print(df_train.iloc[0].to_string())

def set_danger_zone(df):
    tmp = df['in_danger_zone'].copy()
    for i in range(df.shape[0]):
        if df.iloc[i]['CryoSleep'] == 1:
            if df.iloc[i]['Cabin_deck_side'][0] in ['A', 'B', 'C', 'D', 'F']\
                or df.iloc[i]['Cabin_deck_side'] == 'ES' and df.iloc[i]['Cabin_num'] > 320\
                or df.iloc[i]['Cabin_deck_side'] == 'GS' and 620 < df.iloc[i]['Cabin_num'] < 1300:
                tmp.iloc[i] = True
    df['in_danger_zone'] = tmp

# set_danger_zone(df_train)
set_danger_zone(data)
# set_danger_zone(df_test)

# pivot_survival_rate(df_train, 'in_danger_zone')
pivot_survival_rate(data[~data['Transported'].isna()], 'in_danger_zone')
# circle_and_bar(df_train, df_test, 'in_danger_zone')

# print(df_train[df_train['CryoSleep'] == 0].to_string())

# sns.kdeplot(data=df_train[df_train['CryoSleep'] == 0], x="FoodCourt", hue="Transported", common_norm=False)
# plt.xlim(0, df_train['FoodCourt'][df_train['CryoSleep'] == 0].max())
# plt.grid()
# plt.show()

# df_train['total_pay'] = df_train['RoomService'] + df_train['ShoppingMall'] + df_train['Spa'] + df_train['VRDeck'] + df_train['FoodCourt']
# df_test['total_pay'] = df_test['RoomService'] + df_test['ShoppingMall'] + df_test['Spa'] + df_test['VRDeck'] + df_test['FoodCourt']
# print(df_train['total_pay'])
# sns.kdeplot(data=df_train[df_train['CryoSleep'] == 0], x="total_pay", hue="Transported", common_norm=False)
# plt.xlim(0, df_train['total_pay'][df_train['CryoSleep'] == 0].max())
# plt.grid()
# plt.show()
# print(data['total_pay'])
# data['total_pay'] -= data['FoodCourt']
# sns.kdeplot(data=data[data['CryoSleep'] == 0], x="total_pay", hue="Transported", common_norm=False)
# plt.xlim(0, data['total_pay'][data['CryoSleep'] == 0].max())
# plt.grid()
# plt.show()

# df_train['in_crowd_not_food'] = False
data['in_crowd_not_food'] = False
# df_test['in_crowd_not_food'] = False

def set_in_crowd_not_food(df):
    tmp = df['in_crowd_not_food'].copy()
    for i in range(df.shape[0]):
        if df.iloc[i]['CryoSleep'] == 0 and df.iloc[i]['total_pay'] - df.iloc[i]['FoodCourt'] > 1120:
            tmp.iloc[i] = True
    df['in_crowd_not_food'] = tmp

# set_in_crowd_not_food(df_train)
set_in_crowd_not_food(data)
# set_in_crowd_not_food(df_test)

# pivot_survival_rate(df_train[df_train['CryoSleep'] == 0], 'in_crowd_not_food')
pivot_survival_rate(data[(~data['Transported'].isna()) & (data['CryoSleep'] == 0)], 'in_crowd_not_food')
# circle_and_bar(df_train[df_train['CryoSleep'] == 0], df_test[df_test['CryoSleep'] == 0], 'in_crowd_not_food')

# df_train['first_name'] = df_train['Name'].apply(lambda x: x.split()[0] if isinstance(x, str) else np.nan)
# df_train['last_name'] = df_train['Name'].apply(lambda x: x.split()[1] if isinstance(x, str) else np.nan)
# df_test['first_name'] = df_test['Name'].apply(lambda x: x.split()[0] if isinstance(x, str) else np.nan)
# df_test['last_name'] = df_test['Name'].apply(lambda x: x.split()[1] if isinstance(x, str) else np.nan)

# df_train.drop(['Name'], axis=1, inplace=True)
data.drop(['Name', 'Cabin', 'Cabin_deck', 'Cabin_num', 'Cabin_side', 'first_name', 'last_name', 'group', 'group_num', ], axis=1, inplace=True)
# df_test.drop(['Name'], axis=1, inplace=True)

data_train = data[~data['Transported'].isna()]
data_test = data[data['Transported'].isna()]

enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_vars = data_train.dtypes[(data_train.dtypes != "float") & (~data_train.columns.isin(['PassengerId', 'Transported']))].index
print(cat_vars)

# print(data_train.shape, data_test.shape)
ohe_train = pd.DataFrame(enc.fit_transform(data_train[cat_vars].reset_index(drop=True)), columns=enc.get_feature_names_out())
data_train = pd.concat([data_train.reset_index(drop=True), ohe_train], axis=1).drop(cat_vars, axis=1)

ohe_test = pd.DataFrame(enc.transform(data_test[cat_vars].reset_index(drop=True)), columns=enc.get_feature_names_out())
data_test = pd.concat([data_test.reset_index(drop=True), ohe_test], axis=1).drop(cat_vars, axis=1)

# show_high_corr(data_train)

need_to_remove = ['CryoSleep_False', 'in_danger_zone_False', 'in_crowd_not_food_False']
data_train.drop(need_to_remove, axis=1, inplace=True)
data_test.drop(need_to_remove, axis=1, inplace=True)


data = pd.concat([data_train, data_test]).reset_index(drop=True)

# df_train.to_csv('train.csv', index=False)
data.to_csv('data_handled.csv', index=False)
# df_test.to_csv('test.csv', index=False)
