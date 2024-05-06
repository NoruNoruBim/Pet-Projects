import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, make_scorer, f1_score, roc_curve, auc, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def plot_ROC(all_clf, clf_labels):
    colors = ['black', 'orange', 'blue', 'gray', 'green']
    linestyles = [':', '--', '-.', '--', '-']

    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
        # positive is 1 class
        y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)

        plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.3f)' % (label, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()

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


df_train_reduced = pd.read_pickle('df_train_prepared_reduced.pkl')
df_test_reduced = pd.read_pickle('df_test_prepared_reduced.pkl')

y_train_reduced = df_train_reduced['Survived']
x_train_reduced = df_train_reduced.drop(['Survived', 'PassengerId'], axis=1)
x_test_reduced = df_test_reduced.drop(['Survived', 'PassengerId'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x_train_reduced, y_train_reduced, test_size=0.2, random_state=0)

'''
pipe_svc = Pipeline([('scl', ColumnTransformer([('scaler', StandardScaler(), X_train.columns)], remainder='passthrough')),
                    ('clf', SVC(random_state=0))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
              {'clf__C' : [0.7, 0.8, 0.9],
               'clf__kernel' : ['rbf'],
               'clf__gamma' : [0.008, 0.009, 0.01]}
              ]
# param_grid = [{'clf__C' : param_range,
#                'clf__kernel' : ['linear']},
#               {'clf__C' : param_range,
#                'clf__kernel' : ['rbf'],
#                'clf__gamma' : param_range}
#               ]

gs_svc = GridSearchCV(estimator=pipe_svc,
                  scoring='accuracy',
                  param_grid=param_grid,
                  # cv=2,
                  cv=10,
                  n_jobs=-1,
                  verbose=4)

gs_svc = gs_svc.fit(X_train, y_train)

# scores = cross_val_score(gs_svc.best_estimator_, X_train, y_train, scoring='accuracy', cv=5)
scores = cross_val_score(gs_svc.best_estimator_, X_train, y_train, scoring='accuracy', cv=10)
print('Перекрестно-проверочная верность SVM: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print(gs_svc.best_params_)
'''

svc = Pipeline([('scl', ColumnTransformer([('scaler', StandardScaler(), X_train.columns)], remainder='passthrough')),
                    ('clf', SVC(C=0.7, gamma=0.01, kernel='rbf', probability=True, random_state=0))])
svc.fit(X_train, y_train)
scores = cross_val_score(svc, X_train, y_train, scoring='accuracy', cv=10)
print('SVM')
print('Перекрестно-проверочная верность SVM: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('Train accuracy: %.3f' % svc.score(X_train, y_train))
print('Testing accuracy: %.3f' % svc.score(X_test, y_test))
print(f1_score(y_train, svc.predict(X_train)))


# lr = LogisticRegression(C=6, penalty='l2', solver='sag', max_iter=1000)
lr = Pipeline([('scl', ColumnTransformer([('scaler', StandardScaler(), X_train.columns)], remainder='passthrough')),
                    ('clf', LogisticRegression(C=6, penalty='l2', solver='sag', random_state=0, max_iter=1000))])
lr.fit(X_train, y_train)
scores = cross_val_score(lr, X_train, y_train, scoring='accuracy', cv=10)
print('LR')
print('Перекрестно-проверочная верность LR: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('Train accuracy: %.3f' % lr.score(X_train, y_train))
print('Testing accuracy: %.3f' % lr.score(X_test, y_test))
print(f1_score(y_train, lr.predict(X_train)))


rf = RandomForestClassifier(criterion='gini', n_estimators=91, max_depth=4, max_features='sqrt', random_state=0, n_jobs=-1)
# rf = Pipeline([('reducedim', PCA(n_components=5)),
#                     ('clf', RandomForestClassifier(criterion='gini', n_estimators=91, max_depth=4, max_features='sqrt', random_state=0, n_jobs=-1))])
rf.fit(X_train, y_train)
scores = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=10)
print('RF')
print('Перекрестно-проверочная верность LR: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('Train accuracy: %.3f' % rf.score(X_train, y_train))
print('Testing accuracy: %.3f' % rf.score(X_test, y_test))
print(f1_score(y_train, rf.predict(X_train)))


mix = VotingClassifier(estimators=[('svc', svc), ('rf', rf)], voting='soft')
mix.fit(X_train, y_train)
scores = cross_val_score(mix, X_train, y_train, scoring='accuracy', cv=10)
print('MIX')
print('Перекрестно-проверочная верность LR: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('Train accuracy: %.3f' % mix.score(X_train, y_train))
print('Testing accuracy: %.3f' % mix.score(X_test, y_test))
print(f1_score(y_train, mix.predict(X_train)))


tree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0)
ada = AdaBoostClassifier(estimator=tree, random_state=0)

param_grid = [{
               # 'n_estimators' : [200, 220, 240, 260, 280, 300, 320, 340, 360],
               # 'learning_rate' : [0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02],
               # 'n_estimators' : [255, 260, 265],
               # 'learning_rate' : [0.0155, 0.016, 0.0165],
               'n_estimators' : [100, 160, 180, 200, 220],
               'learning_rate' : [0.0005, 0.002, 0.004, 0.006],
              },
              ]

rec_scorer = make_scorer(recall_score, pos_label=1)
gs_ada = GridSearchCV(estimator=ada,
                  # scoring='accuracy',
                  scoring=rec_scorer,
                  param_grid=param_grid,
                  cv=10,
                  n_jobs=-1,
                  verbose=4)

gs_ada = gs_ada.fit(X_train, y_train)
scores = cross_val_score(gs_ada.best_estimator_, X_train, y_train, scoring='accuracy', cv=10)
print(gs_ada.best_params_)
print('ADA')
print('Перекрестно-проверочная верность LR: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('Train accuracy: %.3f' % gs_ada.best_estimator_.score(X_train, y_train))
print('Testing accuracy: %.3f' % gs_ada.best_estimator_.score(X_test, y_test))
print(f1_score(y_train, gs_ada.best_estimator_.predict(X_train)))
print(recall_score(y_train, gs_ada.best_estimator_.predict(X_train), pos_label=0))
print(recall_score(y_train, gs_ada.best_estimator_.predict(X_train), pos_label=1))

# plot_ROC([lr, svc, rf, mix], ['Logistic regression', 'SVM', 'Random forest', 'Voting ensemble'])
plot_ROC([lr, svc, rf, gs_ada.best_estimator_], ['Logistic regression', 'SVM', 'Random forest', 'ADA'])

# rf.fit(x_train_reduced, y_train_reduced)
# print('Final accuracy: %.3f' % rf.score(x_train_reduced, y_train_reduced))
# mix.fit(x_train_reduced, y_train_reduced)
# print('Final accuracy: %.3f' % mix.score(x_train_reduced, y_train_reduced))

# rf.fit(x_train_reduced, y_train_reduced)
# y_pred = rf.predict(x_test_reduced).astype(int)
# df_submission = pd.read_csv("../first_try/gender_submission.csv")
# df_submission['Survived'] = y_pred
# df_submission.to_csv("submission_rf.csv", index=False)
