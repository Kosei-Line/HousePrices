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
Evaluator = Mod.Evaluator
Updater = Mod.Updater

def Load_Dataset():
    #csvからデータを読み取る
    data = pd.read_csv('train.csv')
    #必要なデータだけ取り出す
    #https://www.kaggle.com/code1110/houseprice-data-cleaning-visualization
    data = data[['OverallQual','CentralAir', 'GrLivArea', 'GarageArea',
    'LotArea', 'YearBuilt', 'YearRemodAdd', 'GarageCars', 'BsmtFinSF1',
    'OverallCond', '1stFlrSF', 'BsmtUnfSF', 'GarageType', 'MSZoning',
    'Neighborhood', '2ndFlrSF', 'TotalBsmtSF', 'GarageFinish', 'BsmtQual',
    'OpenPorchSF', 'MoSold', 'BsmtFinType1', 'FireplaceQu', 'KitchenQual',
    'WoodDeckSF', 'SaleCondition', 'Fireplaces', 'MSSubClass', 'EnclosedPorch',
    'SalePrice']]
    data['SalePrice'] = np.log(data['SalePrice'])
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
    #入力データX，教師データT
    X = data[:,:-1].astype('float32')
    T = data[:,-1:].astype('float32')
    #訓練データとテストデータに分ける
    thresh_hold = int(X.shape[0]*0.8)
    train = datasets.TupleDataset(X[:thresh_hold], T[:thresh_hold])
    test  = datasets.TupleDataset(X[thresh_hold:], T[thresh_hold:])
    #訓練データとテストデータを返す
    return train, test

def main():
    #学習ネットワークを持ってくる
    Rec = Net.Rec()
    #gpuを使う
    Rec.to_gpu()
    #データセットの読み込み
    print('Loading dataset')
    train, test = Load_Dataset()
    print('Loaded dataset')

    #make_optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.9, beta2=0.999):
        optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
        return optimizer
    opt = make_optimizer(Rec)
    #set iterator
    train_iter = iterators.SerialIterator(train, args.batch)
    test_iter  = iterators.SerialIterator(test, args.batch,
        repeat=False, shuffle=False)
    #define updater
    updater = Updater.MyUpdater(train_iter, Rec, opt, device=args.gpu)
    #define trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'),
        out="{}/b{}".format(args.out, args.batch))
    #define evaluator
    trainer.extend(Evaluator.MyEvaluator(test_iter, Rec, device=args.gpu))
    #save model
    trainer.extend(extensions.snapshot_object(Rec,
        filename='Rec_epoch_{.updater.epoch}'),
        trigger=(args.snapshot, 'epoch'))
    #out Log
    trainer.extend(extensions.LogReport())
    #print Report
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'val/loss', 'elapsed_time']))
    #display Progress bar
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    del trainer


if __name__ == '__main__':
    main()
    #train, test = Load_Dataset()
