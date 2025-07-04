from tensorflow.keras.datasets import mnist
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Filter only digit '7' from training and test sets
x_train_7 = x_train[y_train == 7]
y_train_7 = y_train[y_train == 7]

x_other_data = x_train[y_train != 7]
y_other_data = y_train[y_train != 7]

x_test_7 = x_test[y_test == 7]
y_test_7 = y_test[y_test == 7]
