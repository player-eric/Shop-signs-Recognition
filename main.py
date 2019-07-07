import numpy as np
from cnn_functions import *
from preprocess import *

training_ratio = 0.7

np.random.seed(int(time.time()))

X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = divide_dataset(training_ratio)
classes = np.arange(1, 61)

X_train = X_train_orig / 255
X_test = X_test_orig / 255
Y_train = convert_to_one_hot(Y_train_orig, 60).T
Y_test = convert_to_one_hot(Y_test_orig, 60).T

parameters = model(X_train, Y_train, X_test, Y_test)
print("finished")
