import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

data_train = pd.read_csv("../data/train.csv")
data_test = pd.read_csv('../data/test.csv')


def show_high_corr(data):
    cols = data.columns[~data.columns.isin(['Id', 'SalePrice'])]
    corr_matrix = data[cols].corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any((upper[column] > 0.9) | (upper[column] < -0.9))]

    need_to_remove = set()
    t = tqdm(total=len(to_drop))
    for element in to_drop:
        column_list = list(data[cols].columns[np.where(
            (data[cols].corrwith(data[cols][element]) > 0.9) |
            (data[cols].corrwith(data[cols][element]) < -0.9))])
        column_list.remove(element)
        need_to_remove.update(map(lambda x: str(x), column_list))
        t.update()
        # for column in column_list:
        #     print(str(element) + " <-> " + str(column) + ": " + str(data[cols][element].corr(data[cols][column])))
    t.close()

    return need_to_remove

# def fill_mode(df, col):
#     return df[col].fillna(df[col].mode().values[0])
#
# def fill_median(df, col):
#     return df[col].fillna(df[col].median())
#
# for col in data_train.loc[:, data_train.dtypes == 'object'].columns:
#     data_train[col] = fill_mode(data_train, col)
# for col in data_train.loc[:, data_train.dtypes != 'object'].drop(['Id', 'SalePrice'], axis=1).columns:
#     data_train[col] = fill_median(data_train, col)
#
# for col in data_test.loc[:, data_test.dtypes == 'object'].columns:
#     data_test[col] = fill_mode(data_test, col)
# for col in data_test.loc[:, data_test.dtypes != 'object'].drop(['Id'], axis=1).columns:
#     data_test[col] = fill_median(data_test, col)

for col in data_train.loc[:, data_train.dtypes == 'object'].columns:
    data_train[col].fillna('No', inplace=True)
for col in data_train.loc[:, data_train.dtypes != 'object'].drop(['Id'], axis=1).columns:
    data_train[col].fillna(0, inplace=True)
for col in data_test.loc[:, data_test.dtypes == 'object'].columns:
    data_test[col].fillna('No', inplace=True)
for col in data_test.loc[:, data_test.dtypes != 'object'].drop(['Id'], axis=1).columns:
    data_test[col].fillna(0, inplace=True)

enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_vars = data_train.dtypes[(data_train.dtypes != "float") & (~data_train.columns.isin(['Id', 'SalePrice']))].index

# print(data_train.shape, data_test.shape)
ohe_train = pd.DataFrame(enc.fit_transform(data_train[cat_vars].reset_index(drop=True)), columns=enc.get_feature_names_out())
data_train = pd.concat([data_train.reset_index(drop=True), ohe_train], axis=1).drop(cat_vars, axis=1)

ohe_test = pd.DataFrame(enc.transform(data_test[cat_vars].reset_index(drop=True)), columns=enc.get_feature_names_out())
data_test = pd.concat([data_test.reset_index(drop=True), ohe_test], axis=1).drop(cat_vars, axis=1)

need_to_remove = show_high_corr(data_train)
print(data_train.shape)
data_train.drop(need_to_remove, axis=1, inplace=True)
data_test.drop(need_to_remove, axis=1, inplace=True)
print(data_train.shape)

data_train.to_csv('train_filled_ohe.csv', index=False)
data_test.to_csv('test_filled_ohe.csv', index=False)
