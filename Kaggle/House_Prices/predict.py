import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data_train = pd.read_csv("train_filled_ohe.csv")
data_test = pd.read_csv("test_filled_ohe.csv")

# forest = RandomForestRegressor(n_estimators=350,
#                                max_depth=20,
#                                min_samples_split=3,
#                                min_samples_leaf=1,
#                                criterion='squared_error', random_state=0, n_jobs=-1)
#
# mlp_pipe = Pipeline([
#                 ('scl', ColumnTransformer([('scaler', StandardScaler(), data_train.drop(['Id', 'SalePrice'], axis=1).columns)], remainder='passthrough')),
#                 ('clf', MLPRegressor(activation='identity', alpha=1.0,
#                                      hidden_layer_sizes=(125,), solver='lbfgs',
#                                      learning_rate='adaptive', max_iter=1000, random_state=0))
#                     ])

grd_boost = GradientBoostingRegressor(learning_rate=0.03,
                                      max_depth=4,
                                      max_features='sqrt',
                                      min_samples_leaf=3,
                                      min_samples_split=2,
                                      n_estimators=950,
                                      subsample=0.8,
                                      random_state=0)

pred = pd.read_csv('../data/sample_submission.csv')
grd_boost.fit(data_train.drop(['Id', 'SalePrice'], axis=1), data_train['SalePrice'].astype('int'))
pred['SalePrice'] = grd_boost.predict(data_test.drop(['Id'], axis=1))
pred.to_csv('predicted.csv', index=False)
