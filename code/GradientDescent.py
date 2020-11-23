import numpy as np
import math
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, learning_rate=0.0279, max_iter=1e4, epsilon=1e-5, batch_size=950,
                 momentum_beta=0.9, lambda_regularization = 0.1, record_history=True):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.beta = momentum_beta
        self.lambda_reg = lambda_regularization
        self.record_history = record_history
        if record_history:
            self.w_history = []
            self.cost_history = []

    def batches(self, x, y, batch_size):
        minis = []
        n_rows = x.shape[0]
        inds = np.random.permutation(n_rows)
        i = 1

        while (i * batch_size) <= n_rows:
            inds_limit = i * batch_size
            start = (i - 1) * batch_size
            x_mini, y_mini = x[inds[start:inds_limit]], y[inds[start:inds_limit]]
            minis.append((x_mini, y_mini))
            i += 1

        else:
            end = (i - 1) * batch_size
            x_left, y_left = x[inds[end:]], y[inds[end:]]
            minis.append((x_left, y_left))
        # print(minis)
        # print('With ', n_rows, 'of samples and a batch size of', batch_size, ': number of batches found =', len(minis))
        return minis

    def cost_fn(self, x, y, w):
        N, D = x.shape
        # N, C = y.shape
        # D, C = w.shape
        N = np.shape(y)
        D = np.shape(w)

        z = np.dot(x, w)  # shape NxC i.e logit
        z_b = np.max(z, axis=1)
        z_bar = z_b[:, None]
        #L2 Regularization variable (Ridge Regression)
        l2_r = self.lambda_reg * (np.square(np.linalg.norm(w)) - np.square(w[0]))
        l2_r = np.mean(l2_r)
        #L1 Regularization variable (Lasso)
        l1_r = self.lambda_reg * np.linalg.norm(w, ord=1)
        #Normal Cost Function (No Regression)
        J = np.mean(-y * z + (np.add(z_bar, np.log(np.exp(z - z_bar)))))
        J = J

        return J


    def split(self, x, y, split_size):
        splits=[]
        n_rows = x.shape[0]
        inds = np.random.permutation(n_rows)
        i = 1
        while (i * split_size) <= n_rows:
            inds_limit = i * split_size
            start = (i -1) * split_size
            x_split, y_split = x[inds[start:inds_limit]] , y[inds[start:inds_limit]]
            splits.append((x_split, y_split))
            i += 1
        else:
            end = (i -1) * split_size
            x_left, y_left = x[inds[end:]], y[inds[end:]]
            splits.append((x_left, y_left))
        return splits

    #gradient descent with crossval for termination condition
    def crossval_run(self, gradient_fn, softmax_fn, encoded, predict, x, y, w, folds=5):
        grad = np.inf
        t = 1
        cv_size = int(x.shape[0]/folds)
        cv_sets = self.split(x, y, cv_size)
        cv_score = []
        cv_T = 20
        res = True
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iter and res == True:
            score = []
            for fold in range(folds):
                x_test, y_test = cv_sets[fold][0], cv_sets[fold][1]

                train_set = [c for d, c in enumerate(cv_sets) if d != fold]
                x_train, y_train = train_set[0][0], train_set[0][1]
                for z in range(1, folds - 1):
                    x_train = np.concatenate((x_train, train_set[z][0]))
                    y_train = np.concatenate((y_train, train_set[z][1]))

                mini_batches = self.batches(x_train, y_train, self.batch_size)
                for batch in mini_batches:
                    x_batch, y_batch = batch
                    if y_batch.shape[0] == 0:
                        continue
                    else:
                        y_batch = y_batch[:,None]
                        y_batch = encoded.transform(y_batch)
                        grad = gradient_fn(x_batch, y_batch, w, softmax_fn)

                        d_w = self.beta * w + (1 - self.beta) * grad
                        w = w - self.learning_rate * d_w
                if (y_batch.shape[0] == 0):
                    pass
                else:

                    cost = self.cost_fn(x_batch, y_batch, w)
                    if self.record_history:
                        self.w_history.append(w)
                        if math.isinf(cost) and cost > 0:
                            self.cost_history.append(10000)
                        elif math.isinf(cost) and cost < 0:
                            self.cost_history.append(-10000)
                        else:
                            self.cost_history.append(cost)
                y_pred = predict(x_test, w)
                y_pred = encoded.inverse_transform(y_pred)
                val_score = accuracy_score(y_test, y_pred)
                score.append(val_score)
            cv_score.append(np.mean(score))
            t += 1
            if (t < cv_T):
                res = all(i < j for i, j in zip(cv_score, cv_score[t:]))
            else:
                res = all(i < j for i, j in zip(cv_score, cv_score[len(cv_score)-cv_T:]))

        plt.plot(self.cost_history)
        plt.grid()
        plt.title('Cross entropy error over iterations for dataset 2')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()
        return w


    def run(self, gradient_fn, softmax_fn, x, y, w, encoded):
        grad = np.inf
        t = 1
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iter:
            mini_batches = self.batches(x, y, self.batch_size)
            for batch in mini_batches:
                x_batch, y_batch = batch
                if y_batch.shape[0] == 0:
                    continue
                else:
                    y_batch = y_batch[:, None]
                    y_batch = encoded.transform(y_batch)
                    grad = gradient_fn(x_batch, y_batch, w, softmax_fn)

                    d_w = self.beta * w + (1 - self.beta) * grad
                    w = w - self.learning_rate * d_w

            cost = self.cost_fn(x_batch, y_batch, w)

            if self.record_history:
                self.w_history.append(w)
                if math.isinf(cost) and cost > 0:
                    self.cost_history.append(10000)
                elif math.isinf(cost) and cost < 0:
                    self.cost_history.append(-10000)
                else:
                    self.cost_history.append(cost)
            t += 1

        #         if t == self.max_iter:
        #             print('Max Iterations reached at ', t, ' Iterations')
        #         else:
        #             print('Number of Iterations: ', t)
        #         print('Gradient Decsent Complete')
        #         print('The cost is:', cost)
        #         print('Momentum: ', self.beta)
        # plotting the cost function
        plt.plot(self.cost_history)
        plt.title('Cross entropy error over iterations for dataset 2')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()
        # print('Batch size:', self.batch_size, 'Beta:', self.beta, 'Alpha:', self.learning_rate)
        return w

