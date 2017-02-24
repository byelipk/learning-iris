# NOTE http://scikit-learn.org/stable/modules/cross_validation.html

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import datasets

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict

from scipy.stats import skew
from scipy.stats import boxcox

from utils.make_split         import *
from utils.check_distribution import *


# Save the best classifier
from sklearn.externals import joblib

softmax_reg_final = joblib.load('softmax_iris.pkl')


iris = datasets.load_iris()
iris_df = pd.DataFrame({
    "sepal_length": iris['data'][:, 0],
    "sepal_width":  iris["data"][:,1],
    "petal_length": iris["data"][:,2],
    "petal_width":  iris["data"][:,3],
    "species":      iris["target"]})

X = iris_df[["petal_length", "petal_width"]]
y = iris_df[["species"]]


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

_, _, X_test, y_test = make_split(split, X, y)

X_test_reshape  = X_test.as_matrix(columns=None)
y_test_reshape  = y_test.as_matrix(columns=None).reshape(-1,)


predictions = softmax_reg_final.predict(X_test_reshape)

n_correct = sum(predictions == y_test_reshape)

print("Results for: Softmax Regression (FINAL)")
print("=======================================")
print("Accuracy:", n_correct / len(y_test_reshape))
print("\n")
