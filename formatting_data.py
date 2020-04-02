import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

#loading the dataset
data = pd.read_csv('creditcard.csv')
fraud = data[data['Class']==1]
valid = data[data['Class']==0]
ratio = len(fraud) / (len(fraud) + len(valid))

#data preprocessing
columns = data.columns.to_list()
columns = [c for c in columns if c not in ["Class"]]
target = "Class"
X = data[columns]
Y = data[target]

classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=ratio),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
                                               contamination=ratio)
}

n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

        # Now thing is these algos will give us a -1 for outlier and +1 for inlier. However, in our dataset, we have 0 for valid(inlier) and 1 for fraud(outlier). So we need to make adjustments to fix this.

    # Reshaping the prediction values
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = 0

    for index in range(len(Y)):
        if y_pred[index] != Y[index]:
            n_errors += 1

    print(clf_name, "resulted in ", n_errors, " errors")
    print("Accuracy for ", clf_name, "is ", accuracy_score(Y, y_pred))
    print("Classification Report for ", clf_name, "- ")
    print(classification_report(Y, y_pred))
