import numpy as np
from sklearn.cross_validation import StratifiedKFold

from predictor import ProteinFunctionPredictor
from utils import load_training_data


if __name__ == "__main__":
    results_str = "Precision: {0:.4f}, recall: {1:.4f}, accuracy: {2:.4f}, f1: {3:.4f}"
    no_folds = 10

    X, y = load_training_data()
    skf = StratifiedKFold(y, no_folds, shuffle=True)
    fold = 1
    f1s = []
    feature_importances = []
    for train, test in skf:
        pfp = ProteinFunctionPredictor()
        pfp.fit(X[train], y[train])
        feature_importances.append(pfp.classifier.feature_importances_)
        precision, recall, accuracy, f1, cm = pfp.score(X[test], y[test])
        f1s.append(f1)
        print("Fold: {0:d}, train set size: {1:d}, test set size: {2:d}".format(fold, len(train), len(test)))
        print(results_str.format(precision, recall, accuracy, f1))
        print(cm)
        fold += 1
    print("Average F1: {0:.4f}".format(np.mean(f1s)))


