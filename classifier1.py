
# 
#   classifier1.py
#   Supervised Learning - Iris Flower
#
#   Created by Mohammed Ataa on 24/08/19.
#   Copyright © 2019 Ataago. All rights reserved.
#   
#   Clasifier - Decision Tree 
#   Collect Training Data -> Train Clasifier -> make Predictions
#   

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[15])

# Test Data
test_idx = [0, 50, 100]
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Training Data
train_target = np.delete(iris.target, test_idx)     # Labels
train_data = np.delete(iris.data, test_idx, axis=0) # Features

# Training Classifier - Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# Prediction
print("Expected:\t", test_target)
print("Prediction:\t", clf.predict(test_data))
