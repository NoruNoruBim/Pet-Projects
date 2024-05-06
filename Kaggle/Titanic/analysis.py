import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

# print(df_train.describe().transpose().to_string())
# print(df_test.describe().transpose().to_string())
# print(df_train.describe(include=['O']).transpose())
# print(df_test.describe(include=['O']).transpose())

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

sns.kdeplot(data=df_train, x="Age", hue="Survived", common_norm=False)
#sns.kdeplot(data=df_train, x="Age", hue="Survived")

plt.xlim(0, df_train['Age'].max())

plt.grid()
plt.show()


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


ax=sns.countplot(data=df_train, x='Pclass', hue='Survived')

for p in ax.patches:
    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

ax.legend(title='Survived', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

pivot_survival_rate(df_train, "Pclass")


sns.kdeplot(data=df_train, x="Fare", hue="Survived", common_norm=False)
plt.grid()
plt.xlim(0, 100)
plt.show()


pivot_survival_rate(df_train, "Sex")

pivot_survival_rate(df_train, "Embarked")


sns.set(font_scale=1.3)
g = sns.catplot(x="Sex", y="Survived", col="Pclass", data=df_train, kind="bar")

for i in range(3):
    ax = g.facet_axis(0, i)

    for c in ax.containers:
        labels = [f'{(v.get_height()):.2f}' for v in c]
        ax.bar_label(c, labels=labels, label_type='center')

plt.show()

g = sns.catplot(x="Survived", y="Age", col="Pclass", data=df_train, kind="swarm")
plt.show()

for feature in ["Sex", "Embarked", "Pclass", "SibSp", "Parch"]:
    g = sns.catplot(x=feature, y="Survived", data=df_train, kind="bar")

    ax = g.facet_axis(0, -1)

    for c in ax.containers:
        labels = [f'{(v.get_height()):.2f}' for v in c]
        ax.bar_label(c, labels=labels, label_type='center')

    plt.show()

for feature in ["Age", "Fare"]:
    g = sns.kdeplot(data=df_train, x=feature, hue="Survived", common_norm=False)
    plt.show()


df_train.to_pickle('df_train.pkl')
df_test.to_pickle('df_test.pkl')
