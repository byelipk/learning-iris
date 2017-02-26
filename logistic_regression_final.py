# NOTE http://scikit-learn.org/stable/modules/cross_validation.html

import numpy as np

from sklearn.externals import joblib

# Load our prepared test data
dataset = np.load("test_set.npy")

X_test_reshape = dataset[()]["X"]
y_test_reshape = dataset[()]["y"]

# Load the classifier and make predictions
log_reg_final = joblib.load('logit_model.pkl')
predictions   = log_reg_final.predict(X_test_reshape)
n_correct     = sum(predictions == y_test_reshape)

print("Results for: Logistic Regression (FINAL)")
print("========================================")
print("Accuracy:", n_correct / len(y_test_reshape))
print("\n")
