import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def plot_ROC(all_clf, clf_labels, X_train, y_train, X_test, y_test):
    colors = ['red', 'orange', 'black', 'blue', 'gray']
    linestyles = ['--', ':', '-.', '--', '-']

    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
        y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)

        plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.4f)' % (label, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    plt.plot([0, 0], [0, 1], color='green', linewidth=1)
    plt.plot([0, 1], [1, 1], color='green', linewidth=1)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.axis('square')
    plt.show()

data_train = pd.read_csv('data_train_prepared.csv', index_col=0)
data_test = pd.read_csv('data_test_prepared.csv', index_col=0)

test_df = pd.read_csv('../data/test_df.csv')
data_test = data_test.set_index('user_id')
data_test = data_test.reindex(index=test_df['user_id']).reset_index()

#   got from feature_selection.py
good_features = ['action_2', 'action_3', 'action_4', 'action_5', 'prev',
                 'current_tariff_tariff_11', 'current_tariff_tariff_15',
                 'current_tariff_tariff_17', 'current_tariff_tariff_25',
                 'current_tariff_tariff_3', 'current_tariff_tariff_4', 'prepayment_True',
                 'cat_action_1_True', 'cat_tariff_high', 'cat_tariff_mid']

y_train = data_train['will_stay'].astype('int')
X_train = data_train.drop(data_train.columns[~data_train.columns.isin(good_features)], axis=1)
X_test = data_test.drop(data_test.columns[~data_test.columns.isin(good_features)], axis=1)

#   got from model_selection.py
mlp = Pipeline([('scl', ColumnTransformer([('scaler', StandardScaler(), X_train.columns)], remainder='passthrough')),
                ('clf', MLPClassifier(alpha=0.0005,
                                      hidden_layer_sizes=110,
                                      learning_rate='invscaling',
                                      learning_rate_init=0.01,
                                      max_iter=1000,
                                      random_state=0))])
mlp.fit(X_train, y_train)

scores = cross_val_score(mlp, X_train, y_train, scoring='roc_auc', cv=10)
print('Cross-validation roc_auc MLP: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\ttrain roc_auc: %.5f' % roc_auc_score(y_train, mlp.predict_proba(X_train)[:, 1]))

plot_ROC([mlp], ['MLP'], X_train, y_train, X_train, y_train)

test_df['proba'] = mlp.predict_proba(X_test)[:, 1]
test_df[['user_id', 'proba']].to_csv('predict_df.csv', index=False)
