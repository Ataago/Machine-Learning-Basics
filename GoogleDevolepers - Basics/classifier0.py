# 
#   clasifier0.py
#   Supervised Learning
#
#   Created by Mohammed Ataa on 24/08/19.
#   Copyright © 2019 Ataago. All rights reserved.
#   
#   Clasifier - Decision Tree from sklearn
#   Collect Training Data -> Train Clasifier -> make Predictions
#   

from sklearn import tree

orange = 1
apple = 0
bumppy = 1
smooth = 0

# Training Data
features = [[140, smooth], [130, smooth], [150, bumppy], [170, bumppy]]     # [100 grams, smooth surface]
labels = [apple, apple, orange, orange]                                     

clf = tree.DecisionTreeClassifier()     # Creating a Decision Tree Object
clf = clf.fit(features, labels)     # Training classifier

print(clf.predict([[149, smooth], [150, bumppy]]))  # Prediction