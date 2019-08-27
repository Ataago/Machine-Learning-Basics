# 
#   classifier2.py
#   Supervised Learning - Dogs
#
#   Created by Mohammed Ataa on 25/08/19.
#   Copyright © 2019 Ataago. All rights reserved.
#   
#   Clasifier - Decision Tree using Numpy for randomness and visulizing with matplot
#   Collect Training Data -> Train Clasifier -> make Predictions
#   

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree

greyhounds = 500    # Red
labrodors = 500     # Blue

greyhound_heights = 28 + 4 * np.random.randn(greyhounds)
labrodor_heights = 24 + 4 * np.random.randn(labrodors)

plt.hist([greyhound_heights, labrodor_heights], stacked=True, color=['r', 'b'])


x = []
y = []
for height in greyhound_heights:
    x.append([height])
    y.append("greyhound")
for height in labrodor_heights:
    x.append([height])
    y.append("labradors")


clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

print(clf.predict([[25]]))
plt.show()