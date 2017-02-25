import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict

# Load our prepared training data
dataset = np.load("training_set.npy")

X_train_reshape = dataset[()]["X"]
y_train_reshape = dataset[()]["y"]


### STEP 6.C ###
#
# Evaluate neural network model.
#
#   - Measure accuracy
#   - Measure confusion matrix
#   - Measure precision
#   - Measure recall
#   - Measure F1
#   - Measure ROC
#
#   NOTE http://scikit-learn.org/stable/modules/cross_validation.html
#
from sklearn.neural_network import MLPClassifier


#########################################
##### GRID SEARCH (Neural Networks) #####
#########################################
#
# NOTE http://scikit-learn.org/stable/modules/model_evaluation.html

print("Running Grid Search...\n")

# STEP 1
# Build the parameter grid.
import itertools
hidden_layer_sizes = []
nurons = np.linspace(1, 7, 7, dtype=np.int)
layers = np.linspace(1, 3, 3, dtype=np.int)

for s in itertools.product(nurons, layers):
    hidden_layer_sizes.append((int(s[0]), int(s[1])))

# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
param_grid = [
    {
        'activation': ['logistic', 'relu', 'tanh'],
        'hidden_layer_sizes': hidden_layer_sizes,
        'solver': ['lbfgs'],
        'alpha': [1e-5],
        'learning_rate': ['constant'],
        'max_iter': [10000],
        'random_state': [42]
    }
]

# STEP 2
# Initialize the model
net = MLPClassifier()

# STEP 3
# Initialize the grid search
grid_search = GridSearchCV(net, param_grid, cv=3, scoring="accuracy")

# STEP 4
# Run grid search
grid_search.fit(X_train_reshape, y_train_reshape)

print("Best Neural Network parameters:", grid_search.best_params_, "\n")


######################################
##### (1) CV: Accuracy (Softmax) #####
######################################

net = grid_search.best_estimator_

scores = cross_val_score(
    net,
    X_train_reshape,
    y_train_reshape,
    scoring="accuracy", cv=3)

##########################################
##### (2) Confusion Matrix (Softmax) #####
##########################################

net = grid_search.best_estimator_

# Returns a numpy array of classifications (i.e. 0,1,2)
y_train_pred = cross_val_predict(
    net,
    X_train_reshape,
    y_train_reshape, cv=3)

# Each row in a the matrix represents an actual class while
# each column represents a predicted class.
conf_matrix = confusion_matrix(y_train_reshape, y_train_pred)

###################################
##### (3) Precision (Softmax) #####
###################################

_precision_score = precision_score(
    y_train_reshape, y_train_pred, average="macro")

################################
##### (4) Recall (Softmax) #####
################################

_recall_score = recall_score(
    y_train_reshape, y_train_pred, average="macro")

##################################
##### (5) F1 Score (Softmax) #####
##################################

_f1_score = f1_score(
    y_train_reshape, y_train_pred, average="macro")

###################################
##### (6) ROC Score (Softmax) #####
##################################


print("Results for: Neureal Network Regression")
print("=======================================")
print("Accuracy:", scores.mean())
print("Precision:", _precision_score)
print("Recall:", _recall_score)
print("F1:", _f1_score)
print("\n")
print("Confusion matrix")
print("----------------")
print(conf_matrix)

print("\n")


# Save the best classifier
from sklearn.externals import joblib
joblib.dump(net, 'net_model.pkl')
