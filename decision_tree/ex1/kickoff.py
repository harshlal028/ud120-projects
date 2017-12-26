import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## DECISION TREE #################################



#### your code goes here
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=50)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

prettyPicture(clf, features_test, labels_test)
plt.show();

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("accuracy: ",acc)

##acc = ### you fill this in!
### be sure to compute the accuracy on the test set


    
def submitAccuracies():
  return {"acc":round(acc,3)}

