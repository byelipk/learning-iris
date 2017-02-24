def make_split(splitter, X_split, y_split):
    # Split out a training set and a test set
    for train_idx, test_idx in splitter.split(X_split, y_split):
        X_2 = X_split.loc[test_idx]
        y_2 = y_split.loc[test_idx]
        X_1 = X_split.loc[train_idx]
        y_1 = y_split.loc[train_idx]
    # Reset indexes
    X_1  = X_1.reset_index(drop=True)
    y_1  = y_1.reset_index(drop=True)
    X_2  = X_2.reset_index(drop=True)
    y_2  = y_2.reset_index(drop=True)
    return (X_1, y_1, X_2, y_2)
