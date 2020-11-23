import numpy as np
from GradientDescent import GradientDescent
from softmax_Regression import softmax_Regression
from sklearn.metrics import accuracy_score
import operator

class hyper_optimization:
    def split(self, x, y, batch_size):
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
        return minis

    def hyper_optimize(self, x, y):
        batches = np.arange(50, x.shape[0], 100)
        learning_rates = np.arange(0.01, 0.1, 0.018)
        beta = np.arange(0.9, 0.99, 0.018)

        folds = 5
        cross_val_size = int(x.shape[0] / folds)
        cross_val_sets = self.split(x, y, cross_val_size)

        score_dict = {}
        counter = 0;
        for i in batches:
            for j in learning_rates:
                for k in beta:
                    score = []
                    for x in range(folds):
                        x_test = cross_val_sets[x][0]
                        y_test = cross_val_sets[x][1]

                        train_set = [c for d, c in enumerate(cross_val_sets) if d != x]
                        x_train = train_set[0][0]
                        y_train = train_set[0][1]
                        for z in range(1, folds - 1):
                            x_train = np.concatenate((x_train, train_set[z][0]))
                            y_train = np.concatenate((y_train, train_set[z][1]))

                        # print(x_train.shape, y_train.shape)
                        gd = GradientDescent(batch_size=i, learning_rate=j, momentum_beta=k)
                        smReg = softmax_Regression()
                        w_opt, enc = smReg.fit(x_train, y_train, gd)
                        y_pred = smReg.predict(x_test, w_opt)
                        y_pred = enc.inverse_transform(y_pred)
                        val_score = accuracy_score(y_test, y_pred)
                        score.append(val_score)
                        counter += 1;
                        print('Iteration:', counter, 'CV Fold:', x, 'Batch size:', i, 'Beta:', k, 'Alpha:', j,
                              'Accuracy:', val_score)

                    score_dict[(i, j, k)] = np.mean(score)

        optimal_hyperp = max(score_dict.items(), key=operator.itemgetter(1))[0]
        print('Optimal hyperparameters, batch_size, learning rate and beta are: \n')
        print(optimal_hyperp)
        print('Average cross_val accuracy score using the optimal values: ', score_dict[optimal_hyperp])
        return optimal_hyperp