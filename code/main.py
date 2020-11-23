from GradientDescent import GradientDescent
from hyper_optimization import hyper_optimization
from loader import Loader
from softmax_Regression import softmax_Regression
from comparison_other_classifier import comparison
import numpy as np
import matplotlib.pyplot as plt
import time

# instantiating classes
loader = Loader()
gd = GradientDescent()
smReg = softmax_Regression()
hyper = hyper_optimization()
compare = comparison()
# obtaining data
x_1, y_1 = loader.numpy_dataset01()
x_2, y_2 = loader.numpy_dataset02()
#Normalizing datasets
x_1_n = loader.min_max_norm(x_1)
y_1_n = loader.min_max_norm(y_1)
x_2_n = loader.min_max_norm(x_2)
y_2_n = loader.min_max_norm(y_2)
#Measuring model runtime
start_time = time.time()
# softmax fit & gradient descent, Hyperparameters returned here
w_opt, enc= smReg.fit(x_1, y_1, gd)
y_pred = smReg.predict(x_1, w_opt)
y_prediction = enc.inverse_transform(y_pred)
#hypter param optimization
optimal_hyperp = hyper.hyper_optimize(x_1, y_1)
# Checking accuracy
smReg.accuracy_check(y_1, y_prediction)
print("--- %s seconds ---" % (time.time() - start_time))
#Comparison against another classifier
compare.kNN_comparison(gd.split, x_2_n, y_2)
compare.tree_comparison(gd.split, x_2_n, y_2)


#Plotting hyper-parameter evolution
batch_size = optimal_hyperp[0]
learning_rate = optimal_hyperp[1]
momentum_beta = optimal_hyperp[2]
param = np.arange(0.9, 0.99, 0.018)


cost_values = []
test_values = []
train_values = []
for i in param:
  gd = GradientDescent(batch_size=batch_size, learning_rate=learning_rate, momentum_beta=i)
  smReg = softmax_Regression()
  w_opt, enc, test_score, train_score, cost = smReg.fit(x_1, y_1, gd)
  test_values.append(test_score)
  train_values.append(train_score)
  cost_values.append(cost)

plt.plot(param, test_values, 'b', label = 'validation accuracy')
plt.plot(param, train_values, 'g', label = 'training accuracy')
plt.plot(param, cost_values, 'r', label = 'cost')
plt.title('Evolution of the cost, training and validation accuracy at varying values of beta')
plt.xlabel('Momentum beta')
plt.legend(loc="best")