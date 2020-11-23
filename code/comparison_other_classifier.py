from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import operator
import numpy as np

class comparison:
    def __init__(self, folds = 5):
        self.folds = folds
        pass


    def test_train(self, x , y , test_size = 0.1):
        return train_test_split(x, y, shuffle=True, test_size=test_size)

# ************************************************ KNN PART *************************************
    def kNN_optimizer(self, splitter,x ,y):
        # Finding the optimal k
        cross_val_size = int(x.shape[0]/self.folds)
        cross_val_sets = splitter(x, y, cross_val_size)
        scores_dict = {}

        values = np.arange(1,25)
        for k in values:
          model = KNeighborsClassifier(n_neighbors=k)
          scores = []
          for x in range(self.folds):
            x_test = cross_val_sets[x][0]
            y_test = cross_val_sets[x][1]

            train_set = [c for d, c in enumerate(cross_val_sets) if d != x]
            x_train = train_set[0][0]
            y_train = train_set[0][1]
            for z in range(1, self.folds-1):
              x_train = np.concatenate((x_train, train_set[z][0]))
              y_train = np.concatenate((y_train, train_set[z][1]))
            model.fit(x_train, y_train)
            fold_accuracy_score = model.score(x_test,y_test)
            scores.append(fold_accuracy_score)

          scores_dict[k] = np.mean(scores)

        optimal_k = max(scores_dict.items(), key=operator.itemgetter(1))[0]
        print('Optimal k using our implemented gridsearch: ', optimal_k)
        #print('Average cross_val accuracy score using the optimal k: ', k_scores_dict[optimal_k])
        return optimal_k

# ************************************************ DECISION TREE PART **********************************
    def dec_tree_optimizer(self, splitter, x, y):
        cross_val_size = int(x.shape[0]/self.folds)
        cross_val_sets = splitter(x, y, cross_val_size)
        # Finding the optimal max_depth
        tree_scores_dict = {}
        k_vals = np.arange(1,25)
        for k in k_vals:
          tree = DecisionTreeClassifier(max_depth=k)
          scores = []
          for x in range(self.folds):
            x_test = cross_val_sets[x][0]
            y_test = cross_val_sets[x][1]

            train_set = [c for d, c in enumerate(cross_val_sets) if d != x]
            x_train = train_set[0][0]
            y_train = train_set[0][1]
            for z in range(1, self.folds-1):
              x_train = np.concatenate((x_train, train_set[z][0]))
              y_train = np.concatenate((y_train, train_set[z][1]))
              #print(x_test.shape, y_test.shape)
              #print(x_train.shape, y_train.shape)
            tree.fit(x_train, y_train)
            fold_accuracy_score = tree.score(x_test,y_test)
            scores.append(fold_accuracy_score)

          tree_scores_dict[k] = np.mean(scores)

        optimal_d = max(tree_scores_dict.items(), key=operator.itemgetter(1))[0]
        print('Optimal max_depth using our implemented gridsearch: ', optimal_d)
        #print('Average cross_val accuracy score using the optimal k: ', k_scores_dict[optimal_k])
        return optimal_d

    def kNN_comparison(self, splitter, x, y):
        optimal_k = self.kNN_optimizer(splitter, x, y)
        xtr, xte, ytr, yte = self.test_train(x, y)
        # PERFORMANCE OF KNN CLASSIFIER ON TEST SET USING THE OPTIMAL k
        knn2 = KNeighborsClassifier(n_neighbors=optimal_k)
        knn2.fit(xtr, ytr)
        y_pred_knn = knn2.predict(xte)
        score_knn = accuracy_score(yte, y_pred_knn)
        print(f"Prediction accuracy of sklearn's KNN using the optimal k obtained: {round(score_knn,3)}")
        return score_knn

    def tree_comparison(self, splitter, x, y):
        optimal_d = self.dec_tree_optimizer(splitter, x, y)
        xtr, xte, ytr, yte = self.test_train(x, y)
        # PERFORMANCE OF DECISION TREE CLASSIFIER ON TEST SET USING THE OPTIMAL Max_depth,
        tree2 = DecisionTreeClassifier(max_depth=optimal_d)
        tree2.fit(xtr, ytr)
        y_pred_tree = tree2.predict(xte)
        score_tree = accuracy_score(yte, y_pred_tree)
        print(f"Prediction accuracy of sklearn's Decision Tree using the optimal max_depth obtained: {round(score_tree,3)}")
        return score_tree
