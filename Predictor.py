import chainer, os
import pandas as pd
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import datasets, iterators, optimizers, serializers, training
from chainer.training import extensions
from sklearn.preprocessing import LabelEncoder

import Mod
args = Mod.args
Net = Mod.Net

def Load_Dataset():
    #csvからデータを読み取る
    data = pd.read_csv('test.csv')
    #必要なデータだけ取り出す
    #https://www.kaggle.com/code1110/houseprice-data-cleaning-visualization
    data = data[['OverallQual','CentralAir', 'GrLivArea', 'GarageArea',
    'LotArea', 'YearBuilt', 'YearRemodAdd', 'GarageCars', 'BsmtFinSF1',
    'OverallCond', '1stFlrSF', 'BsmtUnfSF', 'GarageType', 'MSZoning',
    'Neighborhood', '2ndFlrSF', 'TotalBsmtSF', 'GarageFinish', 'BsmtQual',
    'OpenPorchSF', 'MoSold', 'BsmtFinType1', 'FireplaceQu', 'KitchenQual',
    'WoodDeckSF', 'SaleCondition', 'Fireplaces', 'MSSubClass', 'EnclosedPorch']]
    #objectを数値に変える処理
    #https://qiita.com/katsu1110/items/a1c3185fec39e5629bcb
    for i in range(data.shape[1]):
        if data.iloc[:,i].dtypes == object:
            lbl = LabelEncoder()
            lbl.fit(list(data.iloc[:,i].values))
            data.iloc[:,i] = lbl.transform(list(data.iloc[:,i].values))
    #NaNをなくす
    data = data.fillna(data.median())
    #データを行列に変換
    data = data.as_matrix()
    #テストデータ
    test  = data.astype('float32')
    #テストデータを返す
    return test

def main():
        #学習ネットワークを持ってくる
    Rec = Net.Rec()
    #gpuを使う
    #CLS.to_gpu()
    #データセットの読み込み
    print('Loading dataset')
    test = Load_Dataset()
    print('Loaded dataset')

    a = []
    b = []
    serializers.load_npz('result/b{}/Rec_epoch_{}'.format(args.batch,
        args.epoch), Rec)
    for i in range(len(test)):
        with chainer.using_config('train', False):
            y = Rec(test[[i]]).data[0][0]
        a.append(1461+i)
        b.append(y)
    b = np.exp(b)
    df = pd.DataFrame({
        'Id' : a,
        'SalePrice' : b
    })
    df.to_csv("submit.csv",index=False)

if __name__ == '__main__':
    main()
