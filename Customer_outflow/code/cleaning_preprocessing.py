import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def set_cat_tar(data):
    data['cat_tariff'] = ''
    mid = ['tariff_1', 'tariff_15', 'tariff_2', 'tariff_24', 'tariff_25', 'tariff_26', 'tariff_3', 'tariff_4']
    low = ['tariff_10', 'tariff_12', 'tariff_20']
    high = ['tariff_11', 'tariff_13', 'tariff_14', 'tariff_16', 'tariff_17', 'tariff_21', 'tariff_6', 'tariff_7',
                                                                                                      'tariff_9']
    tmp = data.loc[:, ['cat_tariff']]
    tmp[data['current_tariff'].isin(mid)] = 'mid'
    tmp[data['current_tariff'].isin(low)] = 'low'
    tmp[data['current_tariff'].isin(high)] = 'high'
    data['cat_tariff'] = tmp

def pivot_staying_rate(data, target_column):
    df_pivot = pd.pivot_table(
        data[['user_id', target_column, 'will_stay']],
        index=[target_column],
        columns=["will_stay"],
        aggfunc='count',
        fill_value=0) \
        .reset_index()
    df_pivot.columns = [target_column, 'leaved', 'stayed']
    df_pivot['users'] = df_pivot['stayed'] + df_pivot['leaved']
    df_pivot['stay_rate'] = df_pivot['stayed'] / df_pivot['users'] * 100

    print(df_pivot.to_markdown())
    print()

def catplot(data, feature1, feature2):
    sns.set(font_scale=1.3)
    g = sns.catplot(x=feature1, y="will_stay", col=feature2, data=data, kind="bar")
    for i in range(3):
        ax = g.facet_axis(0, i)
        for c in ax.containers:
            labels = [f'{(v.get_height()):.2f}' for v in c]
            ax.bar_label(c, labels=labels, label_type='center')
    plt.show()

def show_outliers(data_train, data_test):
    num_features = ['action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'prev', 'total']

    data_train['dataset'] = 'train'
    data_test['dataset'] = 'test'
    data = pd.concat([data_train, data_test])

    for feature in num_features:
        print(feature, data[feature].mean(), data[feature].max())
        plt.figure()
        sns.boxplot(data=data, y=feature, x='dataset')
        plt.ylim(0, data[feature].mean() * 3)
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

#   get raw data
data_train = pd.read_csv('main_raw_data_train.csv', index_col=0)
data_test = pd.read_csv('main_raw_data_test.csv', index_col=0)

# for i in list(data_train.columns[1:-3]) + [data_train.columns[-2]]:
#     sns.kdeplot(data=data_train, x=i, hue="will_stay", common_norm=False)
#     plt.title(i)
#     plt.xlim(0, 50)
#     plt.grid()
#     plt.show()

#   discretize features
data_train['cat_action_1'] = data_train['action_1'] >= 9
data_train['cat_action_2'] = data_train['action_2'] >= 13
data_train['cat_action_3'] = data_train['action_3'] >= 5
data_train['cat_action_4'] = data_train['action_4'] >= 2
data_train['cat_action_5'] = data_train['action_5'] >= 1
data_train['cat_prev'] = data_train['prev'] >= 3
data_train['cat_total'] = data_train['total'] >= 29
set_cat_tar(data_train)

data_test['cat_action_1'] = data_test['action_1'] >= 9
data_test['cat_action_2'] = data_test['action_2'] >= 13
data_test['cat_action_3'] = data_test['action_3'] >= 5
data_test['cat_action_4'] = data_test['action_4'] >= 2
data_test['cat_action_5'] = data_test['action_5'] >= 1
data_test['cat_prev'] = data_test['prev'] >= 3
data_test['cat_total'] = data_test['total'] >= 29
set_cat_tar(data_test)

# for i in list(data_train.columns[10:]) + [data_train.columns[7], 'prev', 'prepayment']:
#     print(i)
#     pivot_staying_rate(data_train, i)
#     print()

# catplot(data_train, 'cat_prev', 'cat_tariff')
# show_outliers(data_train, data_test)

#   discretize some features more
# data_train['dataset'] = 'train'
# data_test['dataset'] = 'test'
# data_merged = pd.concat([data_train, data_test])
#
# data_merged['action_2_bin'] = pd.qcut(data_merged['action_2'], 10)
# data_merged['action_3_bin'] = pd.qcut(data_merged['action_3'], 4)
# data_merged['total_bin'] = pd.qcut(data_merged['total'], 10)
# data_merged['prev_bin'] = pd.qcut(data_merged['prev'], 3)
#
# data_train = data_merged[data_merged['dataset'] == 'train'].drop(['dataset'], axis=1)
# data_test = data_merged[data_merged['dataset'] == 'test'].drop(['dataset', 'will_stay'], axis=1)

# fig, axs = plt.subplots()
# sns.countplot(x='prev_bin', hue='will_stay', data=data_train)
# plt.show()

#   decrease skewness
for feature in ['action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'prev', 'total']:
    # df_merge = pd.concat([data_train, data_test])
    # g = sns.displot(df_merge[feature])
    # plt.title("Skewness : %.2f" % (df_merge[feature].skew()))
    # plt.show()

    data_train[feature] = data_train[feature].apply(np.log)
    data_test[feature] = data_test[feature].apply(np.log)

    tmp = data_train.loc[:, [feature]]
    tmp[np.isneginf(data_train[feature])] = 0
    data_train[feature] = tmp
    tmp = data_test.loc[:, [feature]]
    tmp[np.isneginf(data_test[feature])] = 0
    data_test[feature] = tmp

    # df_merge = pd.concat([data_train, data_test])
    # g = sns.displot(df_merge[feature])
    # plt.title("Skewness : %.2f" % (df_merge[feature].skew()))
    # plt.show()


#   encode categorical features
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_vars = data_train.dtypes[(data_train.dtypes != "float") & (~data_train.columns.isin(['user_id', 'will_stay']))].index

# print(data_train.shape, data_test.shape)
ohe_train = pd.DataFrame(enc.fit_transform(data_train[cat_vars].reset_index(drop=True)), columns=enc.get_feature_names_out())
data_train = pd.concat([data_train.reset_index(drop=True), ohe_train], axis=1).drop(cat_vars, axis=1)

ohe_test = pd.DataFrame(enc.transform(data_test[cat_vars].reset_index(drop=True)), columns=enc.get_feature_names_out())
data_test = pd.concat([data_test.reset_index(drop=True), ohe_test], axis=1).drop(cat_vars, axis=1)
# print(data_train.shape, data_test.shape)

#   remove features with high correlation
# show_high_corr(data_train)

need_to_remove = ['total', 'cat_action_2_False', 'cat_action_3_False', 'cat_action_4_False', 'cat_action_5_False',
                 'cat_prev_False', 'cat_total_False', 'prepayment_False', 'cat_action_1_False', 'action_1']
data_train.drop(need_to_remove, axis=1, inplace=True)
data_test.drop(need_to_remove, axis=1, inplace=True)

#   remove features with low variance
# selector = VarianceThreshold(0.05)
# cols = data_train.columns[~data_train.columns.isin(['user_id', 'will_stay'])]
# selector.fit(data_train[cols])
#
# need_to_remove = [i for i in data_train[cols].columns
#                         if i not in data_train[cols].columns[selector.get_support()]]
#
# data_train = data_train.drop(need_to_remove, axis=1)
# data_test = data_test.drop(need_to_remove, axis=1)

# print(data_train.shape)

# plt.figure()
# plt.title('Pearson Correlation of Features', y=1.05, size=25)
# cols = data_train.dtypes[~data_train.columns.isin(['user_id', 'will_stay'])].index
# sns.heatmap(data_train[list(cols)].corr(),linewidths=0.1,vmax=1.0, square=False, linecolor='white', annot=True, cmap="YlGnBu")
# plt.show()

data_train.to_csv('data_train_prepared.csv')
data_test.to_csv('data_test_prepared.csv')
