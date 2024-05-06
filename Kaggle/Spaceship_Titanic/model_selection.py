from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def plot_ROC(all_clf, clf_labels):
    colors = ['black', 'orange', 'blue', 'gray', 'green']
    linestyles = [':', '--', '-.', '--', '-']

    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
        y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)

        plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.4f)' % (label, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.axis('square')
    plt.show()

data = pd.read_csv('data_handled.csv')
data_train = data[~data['Transported'].isna()]
data_test = data[data['Transported'].isna()]

X_train, X_test, y_train, y_test = train_test_split(data_train.drop(['Transported', 'PassengerId'], axis=1),
                                                    data_train['Transported'].astype('int'), test_size=0.2, random_state=0)

'''
ct = ColumnTransformer([('scaler', StandardScaler(), ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'total_pay'])], remainder='passthrough')
ct.fit(X_train)
X_train_std = ct.transform(X_train)
X_test_std = ct.transform(X_test)

# ----------------------------------------------------------------------------
#   LogisticRegression
param_grid = [{
    # 'penalty': ['l1', 'l2', 'elasticnet'],
    # 'C': [10 ** i for i in range(-3, 4)],
    'C': [0.001, 0.01, 0.1, 1., 10.],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}]
gs_lr = GridSearchCV(estimator=LogisticRegression(random_state=0, max_iter=1000),
                  param_grid=param_grid,
                  cv=2,
                  scoring='accuracy',
                  n_jobs=-1,
                  verbose=4)
gs_lr.fit(X_train, y_train)
print('\nLogisticRegression')
print(gs_lr.best_params_)

gs_lr = gs_lr.best_estimator_
scores = cross_val_score(gs_lr, X_train, y_train, scoring='accuracy', cv=5)
print('Cross-validation accuracy LR: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\ttrain roc_auc: %.5f' % roc_auc_score(y_train, gs_lr.predict_proba(X_train)[:, 1]))
print('\ttest roc_auc: %.5f' % roc_auc_score(y_test, gs_lr.predict_proba(X_test)[:, 1]))

# ----------------------------------------------------------------------------
#   SVM
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
              # {'kernel' : ['linear', 'rbf'],
              {'kernel' : ['rbf'],
               'C' : [0.001, 0.01, 0.1, 1., 10.]}
              # {'C' : param_range,
              #  'kernel' : ['linear']},
              # {'C' : param_range,
              #  'kernel' : ['rbf'],
              #  'gamma' : param_range}
              ]
gs_svm = GridSearchCV(estimator=SVC(probability=True, random_state=0),
                  param_grid=param_grid,
                  cv=2,
                  scoring='accuracy',
                  n_jobs=-1,
                  verbose=4)
gs_svm.fit(X_train, y_train)
print('\nSVM')
print(gs_svm.best_params_)

gs_svm = gs_svm.best_estimator_
scores = cross_val_score(gs_svm, X_train, y_train, scoring='accuracy', cv=5)
print('Cross-validation accuracy SVM: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\ttrain roc_auc: %.5f' % roc_auc_score(y_train, gs_svm.predict_proba(X_train)[:, 1]))
print('\ttest roc_auc: %.5f' % roc_auc_score(y_test, gs_svm.predict_proba(X_test)[:, 1]))

# ----------------------------------------------------------------------------
#   RandomForestClassifier
# param_grid = [{
#                'criterion' : ['gini', 'entropy', 'log_loss'],
#                'n_estimators' : [100, 150, 300, 500],
#                # 'max_depth' : [4, 8, 16, 32],
#                # 'max_features' : ['sqrt', 'log2', None]
#               }
#               ]
param_grid = [{
               # 'criterion' : ['gini', 'entropy', 'log_loss'],
               'criterion' : ['entropy'],
               # 'n_estimators' : [1750, 2000, 2250],
               'n_estimators' : [600],
               'max_depth' : [16],
               'min_samples_split' : [2],
               'min_samples_leaf' : [2],
              }
              ]
gs_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=0, n_jobs=-1),
                  param_grid=param_grid,
                  cv=10,
                  scoring='accuracy',
                  n_jobs=-1,
                  verbose=4)

gs_rf.fit(X_train, y_train)
print('\nRandomForestClassifier')
print(gs_rf.best_params_)

gs_rf = gs_rf.best_estimator_
scores = cross_val_score(gs_rf, X_train, y_train, scoring='accuracy', cv=10)
print('Cross-validation accuracy RF: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\ttrain roc_auc: %.5f' % roc_auc_score(y_train, gs_rf.predict_proba(X_train)[:, 1]))
print('\ttest roc_auc: %.5f' % roc_auc_score(y_test, gs_rf.predict_proba(X_test)[:, 1]))

# ----------------------------------------------------------------------------
#   MLP
param_grid = [{
               'hidden_layer_sizes' : [50, 100, 150],
               # 'activation' : ['logistic', 'tanh', 'relu'],
               'solver' : ['lbfgs', 'sgd', 'adam'],
               # 'alpha' : [0.00001, 0.0001, 0.001, 0.01],
               # 'learning_rate' : ['constant', 'invscaling', 'adaptive'],
               # 'learning_rate_init' : [0.0001, 0.001, 0.01],
              }
              ]
gs_mlp = GridSearchCV(estimator=MLPClassifier(max_iter=1000),
                  param_grid=param_grid,
                  cv=2,
                  scoring='accuracy',
                  n_jobs=-1,
                  verbose=4)

gs_mlp.fit(X_train, y_train)
print('\nMLP')
print(gs_mlp.best_params_)

gs_mlp = gs_mlp.best_estimator_
scores = cross_val_score(gs_mlp, X_train, y_train, scoring='accuracy', cv=5)
print('Cross-validation accuracy MLP: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\ttrain roc_auc: %.5f' % roc_auc_score(y_train, gs_mlp.predict_proba(X_train)[:, 1]))
print('\ttest roc_auc: %.5f' % roc_auc_score(y_test, gs_mlp.predict_proba(X_test)[:, 1]))

# ----------------------------------------------------------------------------
#   plot ROC
plot_ROC([gs_lr, gs_svm, gs_rf, gs_mlp], ['LR', 'SVM', 'RF', 'MLP'])
'''

param_grid = [
              {
                'loss' : ['exponential'],
                'learning_rate' : [0.07, 0.075, 0.8],
                'n_estimators' : [350, 400, 450],
                'subsample' : [0.7, 0.75, 0.8],
                'min_samples_split' : [4],
                'min_samples_leaf' : [1],
                'max_depth' : [3],
                'max_features' : [None],
              }
              ]

gs = GridSearchCV(
                  # RandomForestClassifier(random_state=0, n_jobs=-1),
                  GradientBoostingClassifier(random_state=0),
                  param_grid,
                  cv=5, verbose=4, n_jobs=-1)

gs.fit(X_train, y_train)

def print_results(res):
    for param in sorted(zip(res['params'],
                            res['mean_test_score'],
                            res['rank_test_score']),
                        key=lambda x: x[2]):
        print(param[1], param[0], sep='\t')
print_results(gs.cv_results_)

gs = gs.best_estimator_
y_train_pred = gs.predict(X_train)
y_test_pred = gs.predict(X_test)

print('Accuracy train: %.3f, accuracy test: %.3f' % (accuracy_score(y_train, y_train_pred),
                                                     accuracy_score(y_test, y_test_pred)))
