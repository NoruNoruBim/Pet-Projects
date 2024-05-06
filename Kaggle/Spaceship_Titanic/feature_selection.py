import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from time import time

data = pd.read_csv('data_handled.csv')
data_train = data[~data['Transported'].isna()]
data_test = data[data['Transported'].isna()]

X_train, X_test, y_train, y_test = train_test_split(data_train.drop(['Transported', 'PassengerId'], axis=1),
                                                    data_train['Transported'].astype('int'), test_size=0.2, random_state=0)

ct = ColumnTransformer([('scaler', StandardScaler(), ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'total_pay'])], remainder='passthrough')
ct.fit(X_train)
X_train_std = ct.transform(X_train)
X_test_std = ct.transform(X_test)

mlp = MLPClassifier(random_state=0, max_iter=1000)
mlp.fit(X_train_std, y_train)

print('Accuracy train: %.5f' % accuracy_score(y_train, mlp.predict(X_train_std)))
print('Accuracy test: %.5f' % accuracy_score(y_test, mlp.predict(X_test_std)))
print('\ttrain ROC: %.5f' % roc_auc_score(y_train, mlp.predict_proba(X_train_std)[:, 1]))
print('\ttest ROC: %.5f' % roc_auc_score(y_test, mlp.predict_proba(X_test_std)[:, 1]))

print(X_train_std.shape)

for i in range(20, 31, 5):
    start = time()
    print(i)
    mlp = MLPClassifier(random_state=0, max_iter=1000)
    sfs = SequentialFeatureSelector(
           estimator=mlp,
           n_features_to_select=i,
           direction="forward",
           scoring='accuracy',
           n_jobs=-1)
    sfs.fit(X_train_std, y_train)
    mlp.fit(sfs.transform(X_train_std), y_train)

    names = X_train.columns
    print()
    print(f'Selected: {names[sfs.get_support()]}')
    print('Accuracy train: %.5f' % accuracy_score(y_train, mlp.predict(sfs.transform(X_train_std))))
    print('Accuracy test: %.5f' % accuracy_score(y_test, mlp.predict(sfs.transform(X_test_std))))
    print('\ttrain ROC: %.5f' % roc_auc_score(y_train, mlp.predict_proba(sfs.transform(X_train_std))[:, 1]))
    print('\ttest ROC: %.5f' % roc_auc_score(y_test, mlp.predict_proba(sfs.transform(X_test_std))[:, 1]))
    print('time: %d' % (time() - start))
    print()