import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, make_scorer, confusion_matrix, \
    roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC

def plot_conf_matrix(y_train, y_training_pred):
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    confusion_matrix(y_train, y_training_pred).flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         confusion_matrix(y_train, y_training_pred).flatten()/np.sum(confusion_matrix(y_train, y_training_pred))]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_matrix(y_train, y_training_pred), annot=labels, fmt="", cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

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

data_train = pd.read_csv('data_train_prepared.csv', index_col=0)
data_test = pd.read_csv('data_test_prepared.csv', index_col=0)

X_train, X_test, y_train, y_test = train_test_split(data_train.drop(['will_stay', 'user_id'], axis=1),
                                                    data_train['will_stay'].astype('int'), test_size=0.3, random_state=0)

good_features = ['action_2', 'action_3', 'action_4', 'action_5', 'prev',
       'current_tariff_tariff_11', 'current_tariff_tariff_15',
       'current_tariff_tariff_17', 'current_tariff_tariff_25',
       'current_tariff_tariff_3', 'current_tariff_tariff_4', 'prepayment_True',
       'cat_action_1_True', 'cat_tariff_high', 'cat_tariff_mid']

X_train = X_train.drop(X_train.columns[~X_train.columns.isin(good_features)], axis=1)
X_test = X_test.drop(X_test.columns[~X_test.columns.isin(good_features)], axis=1)

ct = ColumnTransformer([('scaler', StandardScaler(), X_train.columns)], remainder='passthrough')
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

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
                  scoring='roc_auc',
                  n_jobs=-1,
                  verbose=4)
gs_lr.fit(X_train, y_train)
print('\nLogisticRegression')
print(gs_lr.best_params_)

gs_lr = gs_lr.best_estimator_
scores = cross_val_score(gs_lr, X_train, y_train, scoring='roc_auc', cv=5)
print('Cross-validation roc_auc LR: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\ttrain roc_auc: %.5f' % roc_auc_score(y_train, gs_lr.predict_proba(X_train)[:, 1]))
print('\ttest roc_auc: %.5f' % roc_auc_score(y_test, gs_lr.predict_proba(X_test)[:, 1]))

# ----------------------------------------------------------------------------
#   SVM
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
              {'kernel' : ['linear', 'rbf'],
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
                  scoring='roc_auc',
                  n_jobs=-1,
                  verbose=4)
gs_svm.fit(X_train, y_train)
print('\nSVM')
print(gs_svm.best_params_)

gs_svm = gs_svm.best_estimator_
scores = cross_val_score(gs_svm, X_train, y_train, scoring='roc_auc', cv=5)
print('Cross-validation roc_auc SVM: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\ttrain roc_auc: %.5f' % roc_auc_score(y_train, gs_svm.predict_proba(X_train)[:, 1]))
print('\ttest roc_auc: %.5f' % roc_auc_score(y_test, gs_svm.predict_proba(X_test)[:, 1]))

# ----------------------------------------------------------------------------
#   RandomForestClassifier
param_grid = [{
               'criterion' : ['gini', 'entropy', 'log_loss'],
               'n_estimators' : [100, 150, 300, 500],
               # 'max_depth' : [4, 8, 16, 32],
               # 'max_features' : ['sqrt', 'log2', None]
              }
              ]
gs_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=0, n_jobs=-1),
                  param_grid=param_grid,
                  cv=2,
                  scoring='roc_auc',
                  n_jobs=-1,
                  verbose=4)

gs_rf.fit(X_train, y_train)
print('\nRandomForestClassifier')
print(gs_rf.best_params_)

gs_rf = gs_rf.best_estimator_
scores = cross_val_score(gs_rf, X_train, y_train, scoring='roc_auc', cv=5)
print('Cross-validation roc_auc RF: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
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
                  scoring='roc_auc',
                  n_jobs=-1,
                  verbose=4)

gs_mlp.fit(X_train, y_train)
print('\nMLP')
print(gs_mlp.best_params_)

gs_mlp = gs_mlp.best_estimator_
scores = cross_val_score(gs_mlp, X_train, y_train, scoring='roc_auc', cv=5)
print('Cross-validation roc_auc MLP: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('\ttrain roc_auc: %.5f' % roc_auc_score(y_train, gs_mlp.predict_proba(X_train)[:, 1]))
print('\ttest roc_auc: %.5f' % roc_auc_score(y_test, gs_mlp.predict_proba(X_test)[:, 1]))

# ----------------------------------------------------------------------------
#   plot ROC
plot_ROC([gs_lr, gs_svm, gs_rf, gs_mlp], ['LR', 'SVM', 'RF', 'MLP'])


# ----------------------------------------------------------------------------
#   MLP and RF final
# param_grid = [{
#                # 'hidden_layer_sizes' : [103, 105, 107],
#                # 'activation' : ['logistic', 'tanh', 'relu'],
#                # 'solver' : ['lbfgs', 'sgd', 'adam'],
#                # 'alpha' : [0.0004, 0.0005, 0.0006],
#                'alpha' : [0.0005],
#                # 'learning_rate' : ['constant', 'invscaling', 'adaptive'],
#                'learning_rate' : ['invscaling'],
#                # 'learning_rate_init' : [0.0001, 0.001, 0.01],
#                'learning_rate_init' : [0.01],
#               }
#               ]
# gs_mlp = GridSearchCV(estimator=MLPClassifier(max_iter=1000),
#                   param_grid=param_grid,
#                   cv=5,
#                   scoring='roc_auc',
#                   n_jobs=-1,
#                   verbose=4)
# gs_mlp.fit(X_train, y_train)
# print('\nMLPClassifier')
# print(gs_mlp.best_params_)
#
# gs_mlp = gs_mlp.best_estimator_
# scores = cross_val_score(gs_mlp, X_train, y_train, scoring='roc_auc', cv=10)
# print('Cross-validation roc_auc MLP: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# print('\ttrain roc_auc: %.5f' % roc_auc_score(y_train, gs_mlp.predict_proba(X_train)[:, 1]))
# print('\ttest roc_auc: %.5f' % roc_auc_score(y_test, gs_mlp.predict_proba(X_test)[:, 1]))
#
# plot_conf_matrix(y_test, gs_mlp.predict(X_test))
# plot_ROC([gs_mlp], ['MLP'])

# param_grid = [{
#                'criterion' : ['gini', 'entropy', 'log_loss'],
#                'n_estimators' : [300, 500, 800],
#                'max_depth' : [4, 8, 16, 32],
#                'max_features' : ['sqrt', 'log2', None]
#               }
#               ]
# param_grid = [{
#                # 'criterion' : ['gini', 'entropy', 'log_loss'],
#                'criterion' : ['entropy'],
#                # 'n_estimators' : [1750, 2000, 2250],
#                'n_estimators' : [1100, 1200, 1300],
#                'max_depth' : [10, 12],
#                'min_samples_split' : [4, 5],
#                'min_samples_leaf' : [1, 2],
#               }
#               ]
# gs_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=0, n_jobs=-1),
#                   param_grid=param_grid,
#                   cv=5,
#                   scoring='roc_auc',
#                   n_jobs=-1,
#                   verbose=4)
# gs_rf.fit(X_train, y_train)
# print('\nRandomForestClassifier')
# print(gs_rf.best_params_)
#
# gs_rf = gs_rf.best_estimator_
# scores = cross_val_score(gs_rf, X_train, y_train, scoring='roc_auc', cv=10)
# print('Cross-validation roc_auc RF: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# print('\ttrain roc_auc: %.5f' % roc_auc_score(y_train, gs_rf.predict_proba(X_train)[:, 1]))
# print('\ttest roc_auc: %.5f' % roc_auc_score(y_test, gs_rf.predict_proba(X_test)[:, 1]))
# plot_ROC([gs_rf], ['RF'])
