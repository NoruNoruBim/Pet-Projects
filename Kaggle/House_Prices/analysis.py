import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# def show_high_corr(data):
#     cols = data.columns[~data.columns.isin(['Id', 'SalePrice'])]
#     corr_matrix = data[cols].corr().abs()
#
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     to_drop = [column for column in upper.columns if any((upper[column] > 0.9) | (upper[column] < -0.9))]
#
#     need_to_remove = set()
#     t = tqdm(total=len(to_drop))
#     for element in to_drop:
#         column_list = list(data[cols].columns[np.where(
#             (data[cols].corrwith(data[cols][element]) > 0.9) |
#             (data[cols].corrwith(data[cols][element]) < -0.9))])
#         column_list.remove(element)
#         need_to_remove.update(map(lambda x: str(x), column_list))
#         t.update()
#         # for column in column_list:
#         #     print(str(element) + " <-> " + str(column) + ": " + str(data[cols][element].corr(data[cols][column])))
#     t.close()

data_train = pd.read_csv("../data/train.csv")
data_test = pd.read_csv('../data/test.csv')

data = pd.concat([data_train, data_test]).reset_index(drop=True)

print(data)

for col in data.loc[:, data.dtypes == 'object'].columns:
    data[col].fillna('No', inplace=True)
for col in data.loc[:, data.dtypes != 'object'].drop(['Id', 'SalePrice'], axis=1).columns:
    data[col].fillna(0, inplace=True)
'''
print(data.isna().sum().to_string())

data['totalBathRooms'] = data['FullBath'] + data['HalfBath'] + data['BsmtFullBath'] + data['BsmtHalfBath']
data['afterRemodelAge'] = data['YrSold'] - data['YearRemodAdd']
# data['landscape'] = data['LotShape'] * data['LandContour']
data['livingArea'] = data['GrLivArea'] / data['GrLivArea'].mean()
# data['newGarage'] = data['GarageQual'] * data['GarageCars']
data['totalSquareFeet'] = data['GrLivArea'] + data['TotalBsmtSF']
data['consolidatedPorch'] = data['WoodDeckSF'] + data['EnclosedPorch'] + data['OpenPorchSF'] + data['3SsnPorch'] + data['ScreenPorch']
data['remodel'] = data['afterRemodelAge'] > 0

sns.regplot(x=data['YearBuilt'], y=data['SalePrice'], line_kws={"color": "red"})
plt.title('YearBuilt')
plt.grid()
plt.show()

sns.regplot(x=data['afterRemodelAge'], y=data['SalePrice'], line_kws={"color": "red"})
plt.title('afterRemodelAge')
plt.grid()
plt.show()

sns.boxplot(x=data['Neighborhood'], y=data['SalePrice'])
plt.title('Neighborhood')
plt.grid()
plt.show()

sns.barplot(x=data['Neighborhood'], y=data['SalePrice'], hue=data['BldgType'], dodge=False)
plt.title('Neighborhood')
plt.grid()
plt.show()

fig, ax = plt.subplots(nrows=2, ncols=2)
# ax[0, 0].set_title('GrLivArea')
ax[0, 0].grid()
sns.scatterplot(ax=ax[0, 0], x=data['GrLivArea'], y=data['SalePrice'])

ax[0, 1].grid()
sns.scatterplot(ax=ax[0, 1], x=data['consolidatedPorch'], y=data['SalePrice'])

ax[1, 0].grid()
sns.scatterplot(ax=ax[1, 0], x=data['PoolArea'], y=data['SalePrice'])

ax[1, 1].grid()
sns.scatterplot(ax=ax[1, 1], x=data['totalSquareFeet'], y=data['SalePrice'])

plt.show()

clr=['gray', 'green', 'blue', 'orange', 'red']
for i in range(5):
    sns.scatterplot(x=data[data['GarageCars'] == i]['GarageArea'], y=data['SalePrice'], color=clr[i])
plt.grid()
plt.legend([0, 1, 2, 3, 4])
plt.show()
'''

plt.title('Pearson Correlation of Features', y=1.05)
corr = data_train.loc[:, data.dtypes != 'object'].drop(['Id'], axis=1).corr()
mask = np.triu(corr)
sns.heatmap(corr,
            mask=mask,
            linewidths=0.1, vmax=1.0, square=False, linecolor='white', annot=True, cmap="YlGnBu",
            xticklabels=True, yticklabels=True)
plt.show()
