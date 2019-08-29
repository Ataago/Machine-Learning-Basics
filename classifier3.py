# 
#   classifier3.py
#   Supervised Learning 
#
#   Created by Mohammed Ataa on 26/08/19.
#   Copyright © 2019 Ataago. All rights reserved.
#   
#   Clasifiers - decision tree and kNeighbours and comaparing accuracy
#   Collect Training Data -> Train Clasifier -> make Predictions
#  

from sklearn import datasets
from sklearn import tree
from sklearn import neighbors
from sklearn.model_selection import train_test_split


# Collecting Data
iris = datasets.load_iris()     # https://en.wikipedia.org/wiki/Iris_flower_data_set
x = iris.data       # Features
y = iris.target     # Labels

print(iris)
# Splitting data using sklearn lib functions
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)


# Two methods for classifiers
def DecisionTreeClassifier():
    """Training Classifier - Decision Tree"""
    clf = tree.DecisionTreeClassifier()
    return clf

def KNeighborsClassifier():
    """Training Classifier - K Neighbors"""
    clf = neighbors.KNeighborsClassifier()
    return clf


# Training Classifier
# clf = DecisionTreeClassifier()
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

# Predictions
predictions = clf.predict(x_test)

# Finding Accuracy
from sklearn.metrics import accuracy_score
print("accuracy: ", accuracy_score(y_test, predictions))