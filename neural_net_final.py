# NOTE http://scikit-learn.org/stable/modules/cross_validation.html

from load_and_prepare_data import *
import numpy as np


# Load our prepared test data
dataset = np.load("test_set.npy")

X_test_reshape = dataset[()]["X"]
y_test_reshape = dataset[()]["y"]


# Load the classifier
from sklearn.externals import joblib
net_final = joblib.load('net_model.pkl')



predictions = net_final.predict(X_test_reshape)

n_correct = sum(predictions == y_test_reshape)

print("Results for: Neural Network (FINAL)")
print("===================================")
print("Accuracy:", n_correct / len(y_test_reshape))
print("\n")
