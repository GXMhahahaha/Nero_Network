# Name : main.py
# Time : 2021/8/4 11:32
import matplotlib.pyplot as plt

from utils import *
import model

train_X, train_Y, test_X, test_Y = load_2D_dataset()
layers_dims = [test_X.shape[0], 20, 3, 1]

para_orig = model.l_layer_model(train_X, train_Y, layers_dims, num_iterations=30000, learning_rate=0.3, grad_check=True)
y_hat_orig = model.predict(test_X, para_orig, len(layers_dims))
acc_orig = model.compute_accuracy(y_hat_orig, test_Y)
print("过拟合的准确度为", acc_orig)
plt.title("model without regularization")
axes = plt.gca()  # gca  -  get_current_axis
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: model.predict(x, para_orig, len(layers_dims)), test_X, test_Y)
print("=======================================================")

para_re = model.l_layer_model_with_re(train_X, train_Y, layers_dims, num_iterations=30000, learning_rate=0.3, lambd=0.7,
                                      grad_check=True)
y_hat_re = model.predict(test_X, para_re, len(layers_dims))
acc_re = model.compute_accuracy(y_hat_re, test_Y)
print("使用L2正则化的准确率为", acc_re)
plt.title("L2_reg")
axes = plt.gca()  # gca  -  get_current_axis
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: model.predict(x, para_re, len(layers_dims)), test_X, test_Y)
print("========================================================")

para_drop = model.l_layer_model_dropout(train_X, train_Y, layers_dims, num_iterations=30000, learning_rate=0.3,
                                        keep_prob=0.8)
y_hat_drop = model.predict(test_X, para_drop, len(layers_dims))
acc_drop = model.compute_accuracy(y_hat_drop, test_Y)
print("使用dropout的准确率为", acc_drop)
plt.title("dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: model.predict(x, para_drop, len(layers_dims)), test_X, test_Y)
