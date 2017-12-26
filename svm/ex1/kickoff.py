from sklearn.svm import SVC
clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

prettyPicture(clf, features_test, labels_test)
plt.show();

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("accuracy: ",acc)
