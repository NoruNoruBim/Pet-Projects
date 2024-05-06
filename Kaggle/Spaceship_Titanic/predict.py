import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('../first_try/data_handled.csv')
data_train = data[~data['Transported'].isna()]
data_test = data[data['Transported'].isna()]

X_train = data_train.drop(['PassengerId', 'Transported'], axis=1)
y_train = data_train['Transported'].astype('int')
X_test = data_test.drop(['PassengerId', 'Transported'], axis=1)

grd_boost = GradientBoostingClassifier(loss='exponential',
                                       learning_rate=0.075,
                                       n_estimators=400,
                                       subsample=0.7,
                                       min_samples_split=4,
                                       min_samples_leaf=1,
                                       max_depth=3,
                                       max_features=None,
                                       random_state=0)
grd_boost.fit(X_train, y_train)

pred = pd.read_csv('../data/sample_submission.csv')
pred['Transported'] = grd_boost.predict(X_test).astype('bool')

pred.to_csv('predicted.csv', index=False)
