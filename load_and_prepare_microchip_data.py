import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("microchips.txt", delimiter=",")

positive = data[np.where(data[:, 2] == 1)]
negative = data[np.where(data[:, 2] == 0)]

# We can see in the plot this dataset is highly nonlinear.
# plt.plot(positive[:, 0], positive[:, 1], "b+", label="Some value")
# plt.plot(negative[:, 0], negative[:, 1], "yo", label="Another value")
# plt.show()


# Create a training and test set
import pandas as pd
from utils.make_split         import *
from utils.check_distribution import *


df = pd.DataFrame({
    "test_1":  data[:,0],
    "test_2":  data[:,1],
    "label":   data[:,2]})

# Should be about 50/50
check_distribution(df, "label")

# This is highly non-linear data. Very low correlation scores.
df.corr()

X = df[["test_1", "test_2"]]
y = df[["label"]]


from sklearn.model_selection  import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

X_train, y_train, X_test, y_test = make_split(split, X, y)

# Reshape our data frames so we can use them in machine learning algorithms.
X_train_reshape  = X_train.as_matrix(columns=None)
y_train_reshape  = y_train.as_matrix(columns=None).reshape(-1,)

# Reshape test data
X_test_reshape  = X_test.as_matrix(columns=None)
y_test_reshape  = y_test.as_matrix(columns=None).reshape(-1,)

###################
# FEATURE MAPPING #
###################
#
# One way to fit the data better is to create more features from each
# data point. A logistic regression classifier trained on this higher-dimension
# feature vector will have a more complex decision boundary and will appear
# nonlinear when drawn in our 2-dimensional plot.
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly  = poly_features.fit_transform(X_train_reshape)
X_test_poly   = poly_features.fit_transform(X_test_reshape)


# Save dataset to disk
training_set = {'X': X_train_poly, 'y': y_train_reshape}
test_set     = {'X': X_test_poly,  'y': y_test_reshape}

print("Saving training and test sets...")

np.save("microchips_train", training_set)
np.save("microchips_test", test_set)
