import chainer
import chainer.functions as F
import chainer.links as L

class Rec(chainer.Chain):
    def __init__(self):
        super(Rec, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(29, 64)
            self.fc2 = L.Linear(None, 1)
            self.bnorm1 = L.BatchNormalization(64)
    def __call__(self, x):
        h = F.relu(self.bnorm1(self.fc1(x)))
        h = self.fc2(h)
        return h