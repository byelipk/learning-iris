import numpy as np

# Load our prepared training data
dataset = np.load("microchips_train.npy")

X_train_poly = dataset[()]["X"]
y_train_reshape = dataset[()]["y"]


#######################
# LOGISTIC REGRESSION #
#######################
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        'multi_class': ['ovr'],
        'solver': ['liblinear'],
        'C': np.linspace(.005, 2.0, 100),
        'penalty': ['l2', 'l1'],
        'max_iter': [10000]
    }
]

log_reg     = LogisticRegression()
grid_search = GridSearchCV(log_reg, param_grid, cv=3, scoring="accuracy")

grid_search.fit(X_train_poly, y_train_reshape)


print("Best Logistic Regression parameters:", grid_search.best_params_, "\n")


from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict


######################################
##### (1) CV: Accuracy (Softmax) #####
######################################

log_reg = grid_search.best_estimator_

scores = cross_val_score(
    log_reg,
    X_train_poly,
    y_train_reshape,
    scoring="accuracy", cv=3)

##########################################
##### (2) Confusion Matrix (Softmax) #####
##########################################

log_reg = grid_search.best_estimator_

# Returns a numpy array of classifications (i.e. 0,1,2)
y_train_pred = cross_val_predict(
    log_reg,
    X_train_poly,
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


# Save the best classifier
from sklearn.externals import joblib
joblib.dump(log_reg, 'log_reg_model.pkl')
