# NOTE http://scikit-learn.org/stable/modules/cross_validation.html

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

### STEP 6.B ###
#
# Evaluate support vector machine model.
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
from sklearn import svm


#################################
##### GRID SEARCH (SVM) #####
#################################
#
# NOTE http://scikit-learn.org/stable/modules/model_evaluation.html

print("Running Grid Search...\n")

# STEP 1
# Build the parameter grid.
param_grid = [
    {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': np.linspace(.05, .5, 100),
    }
]

# STEP 2
# Initialize the model
svm_clf = svm.SVC()

# STEP 3
# Initialize the grid search
grid_search = GridSearchCV(svm_clf, param_grid, cv=3, scoring="accuracy")

# STEP 4
# Run grid search
grid_search.fit(X_train_reshape, y_train_reshape)

print("Best SVM parameters:", grid_search.best_params_, "\n")


##################################
##### (1) CV: Accuracy (SVM) #####
##################################

svm_clf = grid_search.best_estimator_

scores = cross_val_score(
    svm_clf,
    X_train_reshape,
    y_train_reshape,
    scoring="accuracy", cv=3)

######################################
##### (2) Confusion Matrix (SVM) #####
######################################

svm_clf = grid_search.best_estimator_

# Returns a numpy array of classifications (i.e. 0,1,2)
y_train_pred = cross_val_predict(
    svm_clf,
    X_train_reshape,
    y_train_reshape, cv=3)

# Each row in a the matrix represents an actual class while
# each column represents a predicted class.
conf_matrix = confusion_matrix(y_train_reshape, y_train_pred)

###############################
##### (3) Precision (SVM) #####
###############################

_precision_score = precision_score(
    y_train_reshape, y_train_pred, average="macro")

############################
##### (4) Recall (SVM) #####
############################

_recall_score = recall_score(
    y_train_reshape, y_train_pred, average="macro")

##############################
##### (5) F1 Score (SVM) #####
##############################

_f1_score = f1_score(
    y_train_reshape, y_train_pred, average="macro")

###############################
##### (6) ROC Score (SVM) #####
##############################


print("Results for: Support Vector Machine")
print("===================================")
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
joblib.dump(svm_clf, 'svm_model.pkl')
