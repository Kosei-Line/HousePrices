from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import pandas_profiling as pdp

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
DROP_LIST = ['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence',
             'MiscFeature', 'MSSubClass', 'OverallCond', 'KitchenAbvGr', 'EnclosedPorch']
CV = 10
input_dim = 30


def data_info(df, filename):
    # print(df.info())
    df_null = df.isnull().sum()
    # print(df_null[df_null > 0])
    profile = pdp.ProfileReport(df)
    profile.to_file(outputfile=filename)


def scaling(df, median):
    df.fillna(median, inplace=True)
    df = pd.get_dummies(df)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
    df = StandardScaler().fit_transform(df)
    return df


def random_feature(X, y):
    rf = RandomForestRegressor(n_estimators=80, max_features='auto')
    rf.fit(X, y)
    ranking = np.argsort(rf.feature_importances_)
    return ranking


def reg_xgb():
    model = xgb.XGBRegressor()
    param = {'max_depth': [2, 4, 6, 8],
             'n_estimators': [50, 100, 200],
             'learning_rate': [0.1, 0.2, 0.3]
             }
    cv = GridSearchCV(model, param, verbose=3, n_jobs=-1, cv=10)
    return cv


def main():
    df_train = pd.read_csv(TRAIN_DATA_PATH, header=0, index_col=0)
    df_test = pd.read_csv(TEST_DATA_PATH, header=0, index_col=0)

    data_info(df_train, 'train.html')
    data_info(df_test, 'test.html')

    df_train.drop(DROP_LIST, axis=1, inplace=True)
    df_test.drop(DROP_LIST, axis=1, inplace=True)

    for i in range(df_train.shape[1]):
        if df_train.iloc[:, i].dtypes == object:
            lbl = LabelEncoder()
            lbl.fit(list(df_train.iloc[:, i].values) + list(df_test.iloc[:, i].values))
            df_train.iloc[:, i] = lbl.transform(list(df_train.iloc[:, i].values))
            df_test.iloc[:, i] = lbl.transform(list(df_test.iloc[:, i].values))

    x_train = df_train.drop('SalePrice', axis=1)
    x_test = df_test
    xMat = pd.concat([x_train, x_test])
    x_train = scaling(x_train, xMat.median())
    x_test = scaling(x_test, xMat.median())
    y_train = df_train['SalePrice']
    y_train = np.log(y_train)

    print('x train shape:', x_train.shape)
    print('x test shape:', x_test.shape)

    # ranking = random_feature(x_train, y_train)
    # x_train = x_train[:, ranking[:input_dim]]
    # x_test = x_test[:, ranking[:input_dim]]

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

    cv = reg_xgb()
    cv.fit(x_train, y_train, eval_metric='rmse', eval_set=[(x_valid, y_valid)], verbose=False)

    print('Best Score:', cv.best_score_)
    print('Best Estimator:\n', cv.best_estimator_)
    # print('Eval Result:', cv.evals_result())

    y_pred = cv.predict(x_valid)
    print(np.sqrt(mean_squared_error(y_valid, y_pred)))

    y_pred = cv.predict(x_test)
    y_pred = np.exp(y_pred)
    df_pred = pd.DataFrame(y_pred, columns=['SalePrice'])
    df_pred.index = df_pred.index + 1461
    df_test['SalePrice'] = df_pred['SalePrice']
    df_test['SalePrice'].to_csv('submit.csv', index=True)


if __name__ == '__main__':
    main()