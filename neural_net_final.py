# NOTE http://scikit-learn.org/stable/modules/cross_validation.html

from load_and_prepare_data import *


# Save the best classifier
from sklearn.externals import joblib

net_final = joblib.load('neural_net_iris.pkl')
print(net_final)

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


predictions = net_final.predict(X_test_reshape)

n_correct = sum(predictions == y_test_reshape)

print("Results for: Neural Network (FINAL)")
print("===================================")
print("Accuracy:", n_correct / len(y_test_reshape))
print("\n")
