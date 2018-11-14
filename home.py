# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
# %matplotlib inline

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

dataset = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

dataset['Alley'].fillna(value='None', inplace=True)
dataset['BsmtCond'].fillna(value='None', inplace=True)
dataset['BsmtExposure'].fillna(value='None', inplace=True)
dataset['BsmtFinSF1'].fillna(value = 0, inplace = True)
dataset['BsmtFinSF2'].fillna(value = 0, inplace = True)
dataset['BsmtFinType1'].fillna(value='None', inplace=True)
dataset['BsmtFinType2'].fillna(value='None', inplace = True)
dataset['BsmtFullBath'].fillna(value = 0, inplace = True)
dataset['BsmtHalfBath'].fillna(value = 0, inplace = True)
dataset['BsmtQual'].fillna(value = 'None', inplace = True)
dataset['BsmtUnfSF'].fillna(value = 0, inplace = True)
dataset['Electrical'].fillna(value = 'Sbrkr', inplace = True)
dataset['Exterior1st'].fillna(value = 'VinylSd', inplace = True)
dataset['Exterior2nd'].fillna(value = 'VinylSd', inplace = True)
dataset['Fence'].fillna(value = 'None', inplace = True)
dataset['FireplaceQu'].fillna(value = 'None', inplace = True)
dataset['Functional'].fillna(value = 'Typ', inplace = True)
dataset['GarageCars'].fillna(value = 2.0, inplace = True)
dataset['GarageArea'].fillna(value = 519.43, inplace = True)
dataset['GarageCond'].fillna(value = 'None', inplace = True)
dataset['GarageQual'].fillna(value = 'None', inplace = True)
dataset['GarageType'].fillna(value = 'None', inplace = True)
dataset['GarageFinish'].fillna(value = 'None', inplace = True)
dataset['GarageYrBlt'].fillna(value = dataset['YearRemodAdd'], inplace = True)
dataset['KitchenQual'].fillna(value = 'TA', inplace = True)
dataset['LotFrontage'] = dataset.groupby('Neighborhood').transform(lambda x: x.fillna(x.mean())) 
dataset['MSZoning'].fillna(value = 'RL', inplace = True)
dataset.loc[2610 ,['MasVnrType']] = 'BrkFace' 
dataset['MasVnrType'].fillna(value = 'None', inplace = True)
dataset['MasVnrArea'].fillna(value = 0, inplace = True)
dataset['MiscFeature'].fillna(value = 'None', inplace = True)
dataset['PoolQC'].fillna(value = 'None', inplace = True)
dataset['SaleType'].fillna(value = 'WD', inplace = True)
dataset['TotalBsmtSF'].fillna(value = 'None', inplace = True)
dataset.drop(['Utilities', 'Id'], axis = 1, inplace = True)
dataset['TotalBsmtSF']=pd.to_numeric(dataset['TotalBsmtSF'], errors = 'coerce')

labeldict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0 }
labellist = ['BsmtCond', 'BsmtQual', 'ExterCond', 'ExterQual',
            'FireplaceQu', 'GarageCond', 'GarageQual', 'KitchenQual', 'HeatingQC', 'PoolQC']
for i in labellist:
    dataset[i].replace(labeldict, inplace = True)

dataset = pd.get_dummies(dataset)

dataset['SalePrice'] = np.log(dataset['SalePrice'])

from sklearn.preprocessing import StandardScaler
dataset.fillna(value = 0, inplace = True)
scaler = StandardScaler()
scaler.fit_transform(dataset)

train_data = dataset.iloc[0:1460][:]
test_data = dataset.iloc[1460:][:]
test_data = test_data.drop(columns = 'SalePrice')
test_data.loc[test_data['TotalBsmtSF'].isnull()]
test_data['TotalBsmtSF'].fillna(value = 0, inplace = True)

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

Y = train_data['SalePrice']
X_train = train_data.drop(columns = 'SalePrice')
X_train, X_test, Y_, Y_test = train_test_split(X_train, Y, test_size = 0.2)

lasso = Lasso(alpha = 0.01, max_iter = 10000000, normalize = True, tol = 0.0004)
lasso.fit(X_train, Y_)

lasso_coef = lasso.coef_

scores = cross_val_score(lasso, X_test, Y_test, cv = 5)
print(scores.mean())

alphas = [0.0005, 0.001, 0.01, 0.05, 0.1]

def rmsle(regression_model, X, y):
    rmsle =np.sqrt(-(cross_val_score(regression_model, X, y, scoring = 'neg_mean_squared_error', cv = 5)))
    return rmsle

rmsle_result = []
for i in alphas:
    model = Lasso(alpha = i, max_iter = 10000)
    model.fit(X_tr, Y_tr)
    result = rmsle(model, X_dev, Y_dev).mean()
    rmsle_result.append(result)

from sklearn.linear_model import LassoCV

lasso_model = LassoCV(alphas = [0.0005, 0.001, 0.01, 0.05, 0.1], max_iter = 1000000).fit(X_train, Y_)
print(rmsle(lasso_model, X_test, Y_test).mean())

plt.plot(alphas, rmsle_result)
plt.xlabel('Alphas')
plt.ylabel('Root Mean Squared Error')
print(alphas)
print(rmsle_result)

final_lasso = Lasso(alpha = 0.0005, max_iter = 1000000).fit(X_train, Y_)
print(rmsle(final_lasso, X_test, Y_test).mean())

# !pip install eli5
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(final_lasso).fit(X_test, Y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())

features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'GarageArea', 'OverallCond', 'FireplaceQu',
           'YearRemodAdd', 'MSSubClass', 'LotArea', 'ScreenPorch', 'BsmtFinSF1', '1stFlrSF', 
           'TotalBsmtSF']

X_train1 = X_train[features]
X_test1 = X_test[features]
test_data1 = test_data[features]

alphas1=[0.0000001, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001,0.005,0.01,0.05, 0.1, 0.5]
rmsle_result1 = []
for i in alphas_1:
    model = Lasso(alpha = i, max_iter = 100000000)
    model.fit(X_train1, Y_)
    result = rmsle(model, X_test1, Y_test).mean()
    rmsle_result1.append(result)
print(rmsle_result1)

plt.plot(alphas_1, rmsle_result1)
plt.xlabel('Alphas')
plt.ylabel('RMSLE')

lasso_perm = Lasso(alpha = 0.0000001, max_iter = 100000000)
lasso_perm.fit(X_train1, Y_)

y_pred = lasso_perm.predict(test_data1)

print(len(df_test['Id']))
y_pred = np.exp(y_pred)
submission = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': y_pred})
submission.head()
submission.to_csv('lasso.csv', index = False)
