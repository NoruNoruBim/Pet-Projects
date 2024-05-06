from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt


def fill_missing_age(df):
    age_df = df[[
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Family', 'Family_Grouped', 'Title',
        'Prefix', 'TGroup', 'Name']]
    age_train = age_df.loc[(age_df.Age.notnull())]
    age_test = age_df.loc[(age_df.Age.isnull())]

    age_train_y = age_train[['Age']]

    age_train_x = age_train.drop('Age', axis=1)

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Family', 'Family_Grouped', 'Title',
                            'Prefix', 'TGroup', 'Name']
    numeric_features = ['Fare']

    col_transform = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('columnprep', col_transform),
        ('reducedim', TruncatedSVD(n_components=20)),
        ('regression', rtr)
    ])

    pipeline.fit(age_train_x, age_train_y.values.ravel())

    print("R2 score for training set: " + str(pipeline.score(age_train_x, age_train_y)))
    age_train_y_pred = pipeline.predict(age_train_x)

    plt.scatter(age_train_y, age_train_y_pred)
    plt.plot([min(age_train_y_pred), max(age_train_y_pred)], [min(age_train_y_pred), max(age_train_y_pred)], c="red")
    plt.xlabel('Age training set')
    plt.ylabel('Age training set predicted')
    plt.show()

    predictedAges = pipeline.predict(age_test.drop('Age', axis=1))

    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df
