from sklearn.model_selection import StratifiedKFold
from sklean.base import clone

def cross_validate(clf, X, y):

    skfolds = StratifiedKFold(n_splits=3, random_state=42)

    for train_index, test_index in skfolds.split(X, y):
        clone_clf = clone(clf)
        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_folds  = X[test_index]
        y_test_folds  = y[test_index]

        # NOTE This only works for binary classification
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_folds)
        n_correct = sum(y_pred == y_test_folds)
        print(n_correct / len(y_pred))
