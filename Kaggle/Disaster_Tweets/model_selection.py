from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


porter = PorterStemmer()

def tokenizer_stemming(text):
    return [porter.stem(word) for word in text.split()]

data_train = pd.read_csv('train_processed.csv')
data_test = pd.read_csv('test_processed.csv')

data_train.fillna(' ', inplace=True)
data_test.fillna(' ', inplace=True)

# data_train['text'] = data_train['text'] + ' ' + data_train['hashtags']
# data_test['text'] = data_test['text'] + ' ' + data_test['hashtags']

# print(data_train.isna().sum())
# print(data_train.dtypes)

# X_train, X_test, y_train, y_test = train_test_split(data_train[['text', 'hashtags']],
X_train, X_test, y_train, y_test = train_test_split(data_train.drop(['id', 'keyword', 'target'], axis=1),
                                                    data_train['target'].astype('int'),
                                                    test_size=0.2, random_state=0)

pipe = make_column_transformer(
    (TfidfVectorizer(lowercase=False, tokenizer=tokenizer_stemming, ngram_range=(1, 2)), 'text'),
    (TfidfVectorizer(lowercase=False, tokenizer=tokenizer_stemming, ngram_range=(1, 2)), 'hashtags'),
    (TfidfVectorizer(lowercase=False, tokenizer=tokenizer_stemming, ngram_range=(1, 2)), 'location'),
)

rf = Pipeline([
               ('vect', pipe),
               # ('pca', KernelPCA(n_jobs=-1)),
               # ('pca', KernelPCA(kernel='poly', n_components=100, gamma=0.0001, degree=5, n_jobs=-1)),
               # ('clf', RandomForestClassifier(random_state=0, n_jobs=-1))
               ('clf', RandomForestClassifier(criterion='entropy',
                                              max_depth=260,
                                              min_samples_leaf=3,
                                              min_samples_split=2,
                                              n_estimators=300,
                                              random_state=0, n_jobs=-1))
               # ('clf', LogisticRegression(random_state=0))
              ])

param_grid = [
            # {
            #   'pca__n_components' : [75, 100, 125],
            #   'pca__n_components' : [100],
            #   # 'pca__kernel' : ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'],
            #   'pca__kernel' : ['poly'],
            #   'pca__gamma' : [0.000075, 0.0001, 0.00025],
            #   'pca__degree' : [4, 5, 6],
            #   'pca__coef0' : [1, 2, 3],
            #  },
            {
            'clf__criterion': ['entropy'],
            'clf__n_estimators' : [280, 300, 320],
            'clf__max_depth': [240, 260, 280],
            'clf__min_samples_split': [2, 3, 4],
            'clf__min_samples_leaf': [1, 2, 3],
            }
            ]

# gs_rf = GridSearchCV(rf, param_grid, scoring='accuracy', cv=5, verbose=4, n_jobs=-1)
# gs_rf = rf

# gs_rf.fit(X_train, y_train)
print('\nRandomForestClassifier')

# print(gs_rf.best_params_)
def print_results(res):
    for param in sorted(zip(res['params'],
                            res['mean_test_score'],
                            res['rank_test_score']),
                        key=lambda x: x[2]):
        print(param[1], param[0], sep='\t')
# print_results(gs_rf.cv_results_)

# gs_rf = gs_rf.best_estimator_
# scores = cross_val_score(gs_rf, X_train, y_train, scoring='accuracy', cv=10)
# print('Cross-validation accuracy RF: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# print('\ttrain acc: %.5f' % accuracy_score(y_train, gs_rf.predict(X_train)))
# print('\ttest acc: %.5f' % accuracy_score(y_test, gs_rf.predict(X_test)))

# rf.fit(X_train, y_train)
# scores = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=10)
# print('Cross-validation accuracy RF: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# print('\ttrain acc: %.5f' % accuracy_score(y_train, rf.predict(X_train)))
# print('\ttest acc: %.5f' % accuracy_score(y_test, rf.predict(X_test)))

rf.fit(data_train.drop(['id', 'keyword', 'target'], axis=1), data_train['target'].astype('int'))
pred = pd.read_csv('../data/sample_submission.csv')
pred['target'] = rf.predict(data_test.drop(['id', 'keyword'], axis=1)).astype('int')

pred.to_csv('predicted.csv', index=False)

