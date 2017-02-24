import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import datasets

from sklearn.model_selection import StratifiedShuffleSplit

from scipy.stats import skew
from scipy.stats import boxcox

from utils.make_split         import *
from utils.check_distribution import *

### STEP 1 ###
#
# Load our dataset into memory.
#
# According to the dataset description there are:
#
#   - 150 examples split evenly among 3 classes (50 examples each)
#   - All features are recorded in centimeters, so no need for feature scaling
#   - There are no missing values in the dataset.
#
iris = datasets.load_iris()

# print(iris.DESCR)


### STEP 2 ###
#
# Explore the dataset.
#
# Use pandas to perform basic data exploration.
#
#   - Create a data frame
#   - Confirm distribution is 33% across all 3 classes
#   - Confirm "petal_length" and "petal_width" are correlated with the label
#

# Create a Pandas data frame
iris_df = pd.DataFrame({
    "sepal_length": iris['data'][:, 0],
    "sepal_width":  iris["data"][:,1],
    "petal_length": iris["data"][:,2],
    "petal_width":  iris["data"][:,3],
    "species":      iris["target"]})

# Confirm class distribution is 33% for each class. It is!
check_distribution(iris_df, "species")

# Looking at the correlation matrix shows us that "sepal_width" has negative
# correlation across the board. Both "petal_length" and "petal_width" correlate
# highly with the classification of the flower. As the class of the flower
# changes, so do the "petal_length" and "petal_width".
iris_df.corr()


### STEP 3 ###
#
# Feature selection.
#
# The features we're most interested in are:
#
#   - petal_width
#   - petal_length
#
X = iris_df[["petal_length", "petal_width"]]
y = iris_df[["species"]]

### STEP 4 ###
#
# Clean and prepare the data.
#
#   - We don't need to perform feature scaling because all values lie on
#     a similar scale (cm).
#   - Calculate and plot skewness
#

# It seems an acceptable range for skewness is -2/+2. We're OK.
# print(X.skew())


# TODO If we want to reduce skewness we can take the square root.
# np.sqrt(X).hist()
# plt.xlabel("Number of Bins")
# plt.ylabel("Length in cm")
# plt.show()



### STEP 5 ###
#
# Create test, validation, and training set.
#
#   - Use stratified sampling
#   - Can also try to use cross_val_score from sklearn.model_selection
#
#   Stratified Sampling
#   ===================
#
#   Our dataset should be representative of the population we're trying to
#   generalize about. The same holds true for our data splits. Each split we
#   make should be representative of each class (33%).
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


# Split out a training, validation, and test set
X_train, y_train, X_test, y_test = make_split(split, X, y)
# X_train, y_train, X_val, y_val   = make_split(split, X_train, y_train)


# Reshape our data frames so we can use them in machine learning algorithms.
X_train_reshape  = X_train.as_matrix(columns=None)
y_train_reshape  = y_train.as_matrix(columns=None).reshape(-1,)
