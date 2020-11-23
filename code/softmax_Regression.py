import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import warnings

# warnings.filterwarnings("ignore")
class softmax_Regression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias

    def softmax_fn(self, z):
        z = np.exp(z)
        sum_ = np.sum(z, axis=1)
        return z / sum_[:, None]

    def gradient_fn(self, x, y, w, softmax_fn):
        N, D = x.shape
        yh = softmax_fn(np.dot(x, w))
        # print("x shape:", x.shape)
        # print ("w shape:", w.shape)
        grad = np.dot(x.T, yh - y) / N
        return grad

    def one_hot_encode(self, y):
        if y.ndim == 1:
            y = y[:, None]

        encoder = OneHotEncoder(sparse=False)
        return encoder.fit_transform(y), encoder

    def fit(self, x, y, optimizer):
        # If x is 1-D make it 2-D
        if x.ndim == 1:
            x = x[:, None]

        y_enc, encoded = self.one_hot_encode(y)
        if self.add_bias:
            N = x.shape[0]  # First dimension
            x = np.column_stack([x, np.ones(N)])  # Adding 1 column for bias
        N, D = x.shape
        N, C = y_enc.shape
        w = np.zeros((D, C))
        #w_optimal= optimizer.run(self.gradient_fn, self.softmax_fn, x, y, w, encoded)
        w_optimal = optimizer.crossval_run(self.gradient_fn, self.softmax_fn, encoded, self.predict, x, y, w)
        #print('weight: ', w_opt.shape)
        #print('Fitting Complete')
        return w_optimal, encoded

    def predict(self, x, w):
        if x.ndim == 1:
            x = x[:, None]
        nt = x.shape[0]
        # if self.add_bias:
        #     x = np.column_stack([x, np.ones(nt)])
        if x.shape[1] != w.shape[0]:
            x = np.column_stack([x, np.ones(nt)])
        y = np.dot(x,w)
        yh = self.softmax_fn(y)
        yh = (yh == yh.max(axis=1)[:, None]).astype(int)  # assigning 1 to max prob and 0 elsewhere
        #print('Prediction Complete')
        return yh

    def accuracy_check(self, y1, y2):
        print('The accuracy_check is: ', accuracy_score(y1, y2))
