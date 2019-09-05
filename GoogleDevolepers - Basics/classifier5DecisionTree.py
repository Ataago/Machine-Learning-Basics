# 
#   classifier5DecisionTree.py
#   Decision Tree Classification based on CART Algorithm
#
#   Created by Mohammed Ataa on 27/08/19.
#   Copyright © 2019 Ataago. All rights reserved.
#   
#   Clasifier - Decision tree CART
#   Collect Training Data -> Train Clasifier -> make Predictions
#  


########################################################################
#   Decision Tree Types.
#   -Classification tree 
#       analysis is when the predicted outcome is the class (discrete) to which the data belongs.
#
#   -Regression tree  
#       analysis is when the predicted outcome can be considered a real number 
#       (e.g. the price of a house, or a patient's length of stay in a hospital).
#
#   Gini Impurity = 1 - SUMof ( probability_of_labels^2 )
#
#   Information Gain 
#   gain = gini_impurity(root) - (gini_impurity(left) * noRows(left)/noRows(root) + gini_impurity(right) * noRows(right)/noRows(root))
#       for example root has 4 labels (rows)
#
########################################################################




# last column is the training data label
training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 4, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
    ['Yellow', 4, 'Lemon'],
    ['Orange', 4, 'Orange'],
    ['Yellow', 5, 'Orange']
]

testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
        ['Green', 5, 'Orange']
    ]
features_list = ['Color' , 'Diameter', 'Label']

def isNumeric(value):
    """test if a the value is numeric and return bool value"""
    return isinstance(value, int) or isinstance(value, float)

def countOfLabels(rows):
    """Count number of each label occurs in rows and 
    return the dictonary example: {apple -> 1, orange -> 2}
    """

    counts = {}
    for row in rows:
        label = row[-1]     # Label is the last coloumn in training data
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

class Question:
    """Used to partition the dataset based on questions
    to find all possible questions for the given data set"""

    def __init__(self, feature_no, feature_value):
        self.feature_no = feature_no 
        self.feature_value = feature_value

    def __repr__(self):
        """Helper function to print out the class question"""
        condition = "=="
        if isNumeric(self.feature_value):
            condition = ">="
        return "Is %s %s %s" %(features_list[self.feature_no], condition, str(self.feature_value))
    
    def isValid(self, test_row):
        """return boolean value of the question for the test_row"""
        test_feature = test_row[self.feature_no]
        if isNumeric(test_feature):
            return test_feature >= self.feature_value
        else:
            return test_feature == self.feature_value


class Decision_Node:
    """The root node or the internal nodes which asks for a decision Question"""

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Leaf_Node:
    """The leaf node contains the count of the particular labels reached 
    this leaf node from Decision_Nodes.
    """

    def __init__(self, rows):
        self.predictions = countOfLabels(rows)



def partition(rows, question):
    """Parttion the rows (dataset) based on the question"""

    true_rows, false_rows = [], []
    for row in rows:
        if question.isValid(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def giniImpurity(rows):
    """Returns the Impurity value for the labels in the rows"""

    counts = countOfLabels(rows)
    impurity = 1
    for label in counts:
        probability_of_label = counts[label] / float(len(rows))
        impurity -= probability_of_label**2
    return impurity

def informationGain(right, left, root_gini_impurity):
    """Gini impuritiy of root minus the weighted impurity of child nodes"""

    p = float(len(right)) / (len(right) + len(left))
    return root_gini_impurity - (p * giniImpurity(right) + (1 - p) * giniImpurity(left))

def findBestSplit(rows):
    """Return the best Question with its best_information_gain value by
    iterating over each feature
    """

    best_information_gain = 0    # Best Information Gain (higher the better)
    best_question = None     # Best question corresponding to best_information_gain
    root_gini_impurity = giniImpurity(rows)

    no_of_features = len(rows[0]) - 1
    for feature_no in range(no_of_features):
        unique_feature_values = set([row[feature_no] for row in rows])

        for feature_value in unique_feature_values:
            temp_question = Question(feature_no, feature_value)  # create a qustion for each of the possible features with labels

            # For the above question split the data set (partition)
            true_rows, false_rows = partition(rows, temp_question)

            # If the data is not divided 
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            temp_info_gain = informationGain(true_rows, false_rows, root_gini_impurity)
            if temp_info_gain >= best_information_gain:
                best_information_gain, best_question = temp_info_gain, temp_question

    return best_information_gain, best_question

def buildDecisionTree(rows):
    """ decision tree made recursiviely with 
    base condition: when no information gain
    else build right(true_branch) or left tree(flase_branch)
    """

    information_gain, question = findBestSplit(rows)

    if information_gain == 0:   # Base condition where we find the leaf
        return Leaf_Node(rows)

    # Get the partitioned rows for either sides (left and right)
    true_rows, false_rows = partition(rows, question)   
    true_branch = buildDecisionTree(true_rows)
    false_branch = buildDecisionTree(false_rows)

    return Decision_Node(question, true_branch, false_branch)

def printTree(node, spacing = ""):
    """Function to print a Decsiion tree"""

    # If its a leaf
    if isinstance(node, Leaf_Node):
        print(spacing + "Predict", node.predictions)
        return
    
    # Print the Question
    print(spacing + str(node.question))

    # Print the Branches
    print(spacing + "--> True:")
    printTree(node.true_branch, spacing + "\t")
    print(spacing + "--> False:")
    printTree(node.false_branch, spacing + "\t")
    
def classify(row, node):
    """Return the Leaf node for the test row"""
    if isinstance(node, Leaf_Node):
        return node.predictions
    
    if node.question.isValid(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def printLeaf(predictions):
    """Prints the predictions with confidence level"""
    confidence_level = {} # confidence of predictions are captured
    total = sum(predictions.values()) 
    for label in predictions:
        confidence_level[label] = str(int(predictions[label] / total * 100)) + "%"
    return confidence_level

def predict(data, decision_tree):
    """Prediction function which uses the Decision tree"""
    for row in data:
        print("Actual: %s. Predicted: %s" %(row[-1], printLeaf(classify(row, decision_tree))))

if __name__ == '__main__':
    my_tree = buildDecisionTree(training_data)
    printTree(my_tree)
    predict(testing_data, my_tree)