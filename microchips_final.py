# NOTE http://scikit-learn.org/stable/modules/cross_validation.html

import numpy as np


# Load our prepared test data
dataset = np.load("microchips_test.npy")

X_test_poly = dataset[()]["X"]
y_test_reshape = dataset[()]["y"]

# Load the classifier
from sklearn.externals import joblib
log_reg_final = joblib.load('log_reg_model.pkl')

predictions = log_reg_final.predict(X_test_poly)

n_correct = sum(predictions == y_test_reshape)

print("Results for: Logistic Regression (FINAL)")
print("========================================")
print("Accuracy:", n_correct / len(y_test_reshape))
print("\n")
