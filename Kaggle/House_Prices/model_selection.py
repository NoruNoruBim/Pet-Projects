import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data_train = pd.read_csv("train_filled_ohe.csv")
data_test = pd.read_csv('test_filled_ohe.csv')


X_train, X_test, y_train, y_test = train_test_split(data_train.drop(['Id', 'SalePrice'], axis=1),
                                                    data_train['SalePrice'].astype('int'),
                                                    test_size=0.2, random_state=0)

forest = RandomForestRegressor(criterion='squared_error', random_state=0, n_jobs=-1)

mlp_pipe = Pipeline([
                ('scl', ColumnTransformer([('scaler', StandardScaler(), X_train.columns)], remainder='passthrough')),
                ('clf', MLPRegressor(learning_rate='adaptive', max_iter=1000, random_state=0))
                    ])

grd_boost = GradientBoostingRegressor(random_state=0)

param_grid = [
            # {
            # 'n_estimators' : [325, 350],
            # 'max_depth' : [20, 25],
            # 'min_samples_split' : [3],
            # 'min_samples_leaf' : [1],
            # }
            # {
            #     'clf__hidden_layer_sizes' : [(125,)],
            #     'clf__activation' : ['identity'],
            #     'clf__solver' : ['lbfgs'],
            #     'clf__alpha' : [0.0001, 0.01, 1., 100.]
            # }
            {
                'learning_rate' : [0.03],
                'n_estimators' : [850, 900, 950],
                'subsample' : [0.7, 0.8, 0.85],
                'min_samples_split' : [2],
                'min_samples_leaf' : [3],
                'max_depth' : [4],
                'max_features' : ['sqrt'],
            }
            ]

# gs_rf = GridSearchCV(forest, param_grid, scoring='neg_root_mean_squared_log_error', cv=5, verbose=4, n_jobs=-1)
# gs_rf.fit(X_train, y_train)

# gs_mlp = GridSearchCV(mlp_pipe, param_grid, scoring='neg_root_mean_squared_log_error', cv=5, verbose=4, n_jobs=-1)
# gs_mlp.fit(X_train, y_train)
# mlp.fit(X_train, y_train)

gs_grd = GridSearchCV(grd_boost, param_grid, scoring='neg_root_mean_squared_log_error', cv=5, verbose=4, n_jobs=-1)
gs_grd.fit(X_train, y_train)

def print_results(res):
    for param in sorted(zip(res['params'],
                            res['mean_test_score'],
                            res['rank_test_score']),
                        key=lambda x: x[2]):
        print(param[1], param[0], sep='\t')
# print_results(gs_rf.cv_results_)
# print_results(gs_mlp.cv_results_)
print_results(gs_grd.cv_results_)


# gs_rf = gs_rf.best_estimator_
# y_train_pred = gs_rf.predict(X_train)
# y_test_pred = gs_rf.predict(X_test)

# gs_mlp = gs_mlp.best_estimator_
# y_train_pred = gs_mlp.predict(X_train)
# y_test_pred = gs_mlp.predict(X_test)

# y_train_pred = mlp.predict(X_train)
# y_test_pred = mlp.predict(X_test)

gs_grd = gs_grd.best_estimator_
y_train_pred = gs_grd.predict(X_train)
y_test_pred = gs_grd.predict(X_test)

#   root mean squared log error
print('RMSLE train: %.3f, RMSLE test: %.3f' % (root_mean_squared_log_error(y_train, y_train_pred),
                                               root_mean_squared_log_error(y_test, y_test_pred)))

print('R^2 train: %.3f, R^2 test: %.3f' % (r2_score(y_train, y_train_pred),
                                           r2_score(y_test, y_test_pred)))

