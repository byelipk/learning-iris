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
training_set = np.load("training_set.npy")

X_train_reshape = training_set[()]["X"]
y_train_reshape = training_set[()]["y"]


### STEP 6.A ###
#
# Evaluate softmax regression model.
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
from sklearn.linear_model    import LogisticRegression


#################################
##### GRID SEARCH (Softmax) #####
#################################
#
# NOTE http://scikit-learn.org/stable/modules/model_evaluation.html

print("Running Grid Search...\n")

# STEP 1
# Build the parameter grid. We want to run softmax regression with 11
# different values for the regulation parameter 'C'.
param_grid = [
    {
        'multi_class': ['ovr'],
        'solver': ['lbfgs', 'newton-cg', 'liblinear'],
        'C': np.linspace(.005, .2, 100),
        'max_iter': [10000],
        'penalty': ['l2']
    }
]

# STEP 2
# Initialize the model
log_reg = LogisticRegression()

# STEP 3
# Initialize the grid search
grid_search = GridSearchCV(log_reg, param_grid, cv=3, scoring="accuracy")

# STEP 4
# Run grid search
grid_search.fit(X_train_reshape, y_train_reshape)

print("Best Logistic Regression parameters:", grid_search.best_params_, "\n")


######################################
##### (1) CV: Accuracy (Softmax) #####
######################################

softmax_reg = grid_search.best_estimator_

scores = cross_val_score(
    log_reg,
    X_train_reshape,
    y_train_reshape,
    scoring="accuracy", cv=3)

##########################################
##### (2) Confusion Matrix (Softmax) #####
##########################################

log_reg = grid_search.best_estimator_

# Returns a numpy array of classifications (i.e. 0,1,2)
y_train_pred = cross_val_predict(
    log_reg,
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


print("Results for: Logistic Regression")
print("================================")
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
joblib.dump(softmax_reg, 'logit_model.pkl')
