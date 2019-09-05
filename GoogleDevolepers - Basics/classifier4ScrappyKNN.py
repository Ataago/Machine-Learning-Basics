# 
#   classifier4ScrappyKNN.py
#   Supervised Learning 
#
#   Created by Mohammed Ataa on 26/08/19.
#   Copyright © 2019 Ataago. All rights reserved.
#   
#   Clasifier - implementation of KNeighbours with k = 1
#   Collect Training Data -> Train Clasifier -> make Predictions
#  

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance 

class ScrappyKNN():
    def fit(self, x_train, y_train):
        """ ScrappyKNN trainer """
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self, x_test):
        """ Prediction for test data """
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        """ closest 1 neighbour as k = 1 """
        shortest_dist = distance.euclidean(row, self.x_train[0])
        shortest_dist_index = 0

        for i in range(1, len(self.x_train)):
            dist = distance.euclidean(row, self.x_train[i])
            if dist < shortest_dist:
                shortest_dist = dist
                shortest_dist_index = i
        return self.y_train[shortest_dist_index]

# Collecting Data
iris = datasets.load_iris()
x = iris.data       # Features
y = iris.target     # Labels

# Splitting data using sklearn lib functions
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# Training Classifier
clf = ScrappyKNN()
clf.fit(x_train, y_train)

# Predictions
predictions = clf.predict(x_test)

# Finding Accuracy
print("accuracy: ", accuracy_score(y_test, predictions))