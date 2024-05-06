import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.stats import skew
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold


df_train = pd.read_pickle('df_train.pkl')
df_test = pd.read_pickle('df_test.pkl')

def pivot_survival_rate(df_train, target_column):
    df_pivot = pd.pivot_table(
        df_train[['PassengerId', target_column, 'Survived']],
        index=[target_column],
        columns=["Survived"],
        aggfunc='count',
        fill_value=0)\
        .reset_index()

    df_pivot.columns = [target_column, 'not_survived', 'survived']

    df_pivot['passengers'] = df_pivot['not_survived']+df_pivot['survived']

    df_pivot['survival_rate'] = df_pivot['survived']/df_pivot['passengers']*100

    print(df_pivot.to_markdown())
    print()
    return df_pivot


def get_family(df):
    df['Family'] = df["Parch"] + df["SibSp"] + 1

    family_map = {
        1: 'Alone',
        2: 'Small',
        3: 'Small',
        4: 'Small',
        5: 'Medium',
        6: 'Medium',
        7: 'Large',
        8: 'Large',
        9: 'Large',
        10: 'Large',
        11: 'Large'}
    df['Family_Grouped'] = df['Family'].map(family_map)

    return df


df_train = get_family(df_train)
df_test = get_family(df_test)

pivot_survival_rate(df_train, "Family_Grouped")


df_train["dataset"] = "train"
df_test["dataset"] = "test"
df_merge = pd.concat([df_train, df_test])
# print(df_merge[["Name"]].values[:5])

def get_name_information(df):
    df[['Last','First']] = df['Name'].str.split(",", n=1, expand=True)
    df[['Title','First']] = df['First'].str.split(".", n=1, expand=True)

    df['Title'] = df['Title'].str.replace(' ', '')

    return df

df_merge = get_name_information(df_merge)

last_pivot = pd.pivot_table(
    df_merge,
    values='Name',
    index='Last',
    columns='dataset',
    aggfunc='count'
    )\
    .sort_values(['train'], ascending=False)

# print(last_pivot)
# print("Number of families that are only in the training data: " + str(len(last_pivot[last_pivot.test.isnull()])))
# print("Number of families that are only in the test data: " + str(len(last_pivot[last_pivot.train.isnull()])))
# print("Number of families that are in the training and test data: " + str(len(last_pivot[last_pivot.notnull().all(axis=1)])))

df_train = get_name_information(df_train)
df_test = get_name_information(df_test)

# print(df_train['Title'].value_counts())

def rename_title(df):
    df.loc[df["Title"] == "Dr", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Rev", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Col", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Ms", "Title"] = 'Miss'
    df.loc[df["Title"] == "Major", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Mlle", "Title"] = 'Miss'
    df.loc[df["Title"] == "Mme", "Title"] = 'Mrs'
    df.loc[df["Title"] == "Jonkheer", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Lady", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "theCountess", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Capt", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Don", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Sir", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Dona", "Title"] = 'Rare Title'
    return df

df_train = rename_title(df_train)
df_test = rename_title(df_test)

pivot_survival_rate(df_train, "Title")


def get_prefix(ticket):
    lead = ticket.split(' ')[0][0]
    if lead.isalpha():
        return ticket.split(' ')[0]
    else:
        return 'NoPrefix'

def ticket_features(df):
    df['Ticket'] = df['Ticket'].replace('LINE','LINE 0')
    df['Ticket'] = df['Ticket'].apply(lambda x: x.replace('.','').replace('/','').lower())
    df['Prefix'] = df['Ticket'].apply(lambda x: get_prefix(x))
    df['TNumeric'] = df['Ticket'].apply(lambda x: int(x.split(' ')[-1]))
    df['TNlen'] = df['TNumeric'].apply(lambda x : len(str(x)))
    df['LeadingDigit'] = df['TNumeric'].apply(lambda x : int(str(x)[0]))
    df['TGroup'] = df['TNumeric'].apply(lambda x: str(x//10))

    df = df.drop(columns=['Ticket','TNumeric'])

    return df

df_train = ticket_features(df_train)
df_test = ticket_features(df_test)

pivot_survival_rate(df_train, "LeadingDigit")

pivot_survival_rate(df_train, "TNlen")

pivot_survival_rate(df_train, "Prefix")

df_tgroup = df_train[['TGroup', 'PassengerId']].groupby(['TGroup']).nunique().sort_values(by='PassengerId', ascending=False)
df_tgroup = df_tgroup[df_tgroup.PassengerId >= 10]

df_tgroup = df_train[df_train['TGroup'].isin(df_tgroup.index)]

sns.catplot(x='TGroup', y="Survived", data=df_tgroup, kind="bar")
plt.show()

num_features = ["Age", "Fare"]

df_merge = pd.concat([df_train, df_test])

for feature in num_features:
    plt.figure()
    sns.boxplot(data=df_merge, y=feature, x="dataset")
    plt.show()

def find_missing_values(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data = missing_data[missing_data['Total']>0]
    return missing_data

print(find_missing_values(df_train))
print(find_missing_values(df_test), end='\n\n')

df_train = df_train.drop(['Cabin'], axis=1)
df_test = df_test.drop(['Cabin'], axis=1)

print(df_train[df_train['Embarked'].isnull()].to_string(), end='\n\n')
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=df_train[df_train['Fare']<200])
plt.show()

df_train["Embarked"] = df_train["Embarked"].fillna("C")

print(df_test[df_test['Fare'].isnull()].to_string())
median_fare = df_test[(df_test['Pclass'] == 3) & (df_test['Embarked'] == 'S')]['Fare'].median()
df_test["Fare"] = df_test["Fare"].fillna(median_fare)


train_objs_num = len(df_train)

features_all = pd.concat([df_train, df_test], axis=0)
from predict_age import fill_missing_age
features_all = fill_missing_age(features_all)

df_train = features_all[:train_objs_num]
df_test = features_all[train_objs_num:]

def age_category(row):
    if row < 12:
        return 'children'
    if (row >= 12) & (row < 60):
        return 'adult'
    if row >= 60:
        return 'senior'
    else:
        return 'no age'

df_train['Age_category'] = df_train['Age'].apply(lambda row: age_category(row))
df_test['Age_category'] = df_test['Age'].apply(lambda row: age_category(row))

pivot_survival_rate(df_train, "Age_category")


df_train['Fare_bin'] = pd.qcut(df_train['Fare'], 13)

fig, axs = plt.subplots(figsize=(20, 8))
sns.countplot(x='Fare_bin', hue='Survived', data=df_train)
plt.show()

df_train['Fare_bin'] = pd.qcut(df_train['Fare'], 13, labels=False)
df_test['Fare_bin'] = pd.qcut(df_test['Fare'], 13, labels=False)

(df_train
    .groupby("Fare_bin")["Survived"]
    .value_counts(normalize=True)
    .mul(100)
    .rename('percent')
    .reset_index()
    .pipe((sns.catplot,'data'), x="Fare_bin", y='percent', hue="Survived", kind='bar'))
plt.axhline(y=38, color='g', linestyle='-')
plt.show()

df_train['Age_bin'] = pd.qcut(df_train['Age'], 10, labels=False)
df_test['Age_bin'] = pd.qcut(df_test['Age'], 10, labels=False)

(df_train
    .groupby("Age_bin")["Survived"]
    .value_counts(normalize=True)
    .mul(100)
    .rename('percent')
    .reset_index()
    .pipe((sns.catplot,'data'), x="Age_bin", y='percent', hue="Survived", kind='bar'))
plt.axhline(y=38, color='g', linestyle='-')
plt.show()

df_merge = pd.concat([df_train, df_test])

for feature in df_merge.columns:
    if df_merge[feature].nunique() < 100:
        sns.displot(df_merge, x=feature, hue="dataset", multiple="stack")
        plt.show()


def compute_skewed_features(df):
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    skewed_feats = pd.DataFrame(index=numeric_feats, columns=['skewness', 'unique_values'])
    skewed_feats['skewness'] = df[numeric_feats].apply(lambda x: skew(x))
    skewed_feats['unique_values'] = df.nunique()
    skewed_feats = skewed_feats[(skewed_feats['skewness'] > 1) | (skewed_feats['skewness'] < -1)]

    return skewed_feats

df_merge = pd.concat([df_train, df_test])
skewed_feats = compute_skewed_features(df_merge)

print(skewed_feats)

for feature in ['Fare']:
    df_merge = pd.concat([df_train, df_test])
    g = sns.displot(df_merge[feature])
    plt.title("Skewness : %.2f" % (df_merge[feature].skew()))
    plt.show()

    df_train[feature] = df_train[feature].apply(np.log)
    df_test[feature] = df_test[feature].apply(np.log)

    df_train[feature][np.isneginf(df_train[feature])] = 0
    df_test[feature][np.isneginf(df_test[feature])] = 0

    df_merge = pd.concat([df_train, df_test])
    g = sns.displot(df_merge[feature])
    plt.title("Skewness : %.2f" % (df_merge[feature].skew()))
    plt.show()

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
cat_vars = df_train.dtypes[df_train.dtypes == "object"].index

ohe_train = pd.DataFrame(enc.fit_transform(df_train[cat_vars]), columns=enc.get_feature_names_out())
print("df_train old shape: " + str(df_train.shape))
print("ohe_train old shape: " + str(ohe_train.shape))
df_train = pd.concat([df_train, ohe_train], axis=1).drop(cat_vars, axis=1)
print("df_train new shape: " + str(df_train.shape))

ohe_test = pd.DataFrame(enc.transform(df_test[cat_vars]), columns=enc.get_feature_names_out())
print("df_test old shape: " + str(df_test.shape))
print("ohe_test old shape: " + str(ohe_test.shape))
df_test = pd.concat([df_test, ohe_test], axis=1).drop(cat_vars, axis=1)
print("df_test new shape: " + str(df_test.shape))

df_train.to_pickle('df_train_prepared.pkl')
df_test.to_pickle('df_test_prepared.pkl')


selector = VarianceThreshold(0.05)
selector.fit(df_train)

need_to_remove = [i for i in df_train.columns if i not in df_train.columns[selector.get_support()]]

df_train = df_train.drop(need_to_remove, axis=1)
df_test = df_test.drop(need_to_remove, axis=1)


corr_matrix = df_train.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any((upper[column] > 0.9) | (upper[column] < -0.9))]

for element in to_drop:
    column_list = list(df_train.columns[np.where(
        (df_train.corrwith(df_train[element]) > 0.9) |
        (df_train.corrwith(df_train[element]) < -0.9))])
    column_list.remove(element)
    for column in column_list:
        print(str(element) + " <-> " + str(column) + ": " + str(df_train[element].corr(df_train[column])))

df_train.drop(['Fare', 'Age', 'Sex_male'], axis=1, inplace=True)
df_test.drop(['Fare',  'Age', 'Sex_male'], axis=1, inplace=True)

plt.figure(figsize=(16,12))
plt.title('Pearson Correlation of Features', y=1.05, size=25)
sns.heatmap(df_train.corr(),linewidths=0.1,vmax=1.0, square=False, linecolor='white', annot=True, cmap="YlGnBu")
plt.show()

df_train.to_pickle('df_train_prepared_reduced.pkl')
df_test.to_pickle('df_test_prepared_reduced.pkl')
