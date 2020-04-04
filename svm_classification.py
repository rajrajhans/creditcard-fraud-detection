from preprocessing import preprocessor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC

features_train, features_test, labels_train, labels_test = preprocessor()

clf = SVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print("Accuracy = ", accuracy_score(labels_test, pred))
print()
print(classification_report(labels_test, pred))
