'''
15.05.2024
Decision Tree Classifier from Scratch
Pema Gurung
Selman Aydin
Wen Wen
'''

import numpy as np
from collections import Counter
import numpy as np

class TreeNode():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain):
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.left = None
        self.right = None

class DecisionTree():
    """
    Decision Tree Classifier
    """

    def __init__(self, max_depth=4, min_samples_leaf=1,min_information_gain=0.0):
        """
        Setting the hyperparameters
        max_depth: (int) -> max depth of the tree
        min_samples_leaf: (int) -> min # of samples required to be in a leaf to make the splitting possible
        min_information_gain: (float) -> min information gain required to make the splitting possible                                                 
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain

    def calculate_entropy(self, class_probabilities):
        """
        Calculate the entropy fort the given class probabilities
        class_probabilities: (list) -> list of class probabilities
        Returns entropy
        """
        entropy = 0
        for probability in class_probabilities:
            if probability > 0:
                entropy += probability * np.log2(probability)
        return -entropy
    
    def calculate_label_probabilities(self, labels, type='entropy'):
        """
        Calculate the label probabilities for the given labels
        labels: (list) -> list of labels
        Returns label probabilities
        """
        if type == 'entropy':
            total_count = len(labels)
            count_labels = Counter(labels).values()
            return [label_count / total_count for label_count in count_labels]
        else:
            total_count = len(labels)
            count_labels = np.zeros(len(self.train_labels), dtype=float)
            for label in labels.astype(int):
                count_labels[label] += 1    
            #count_labels = Counter(labels).values()
            return [label_count / total_count for label_count in count_labels]

    def partition_entropy(self, lower_side_labels, higher_side_labels):
        """
        Calculate the partition entropy for the given lower and higher side labels
        lower_side_labels: (list) -> list of lower side labels
        higher_side_labels: (list) -> list of higher side labels
        Returns partition entropy
        """
        total_count = len(lower_side_labels) + len(higher_side_labels)

        lower_side_probs = self.calculate_label_probabilities(lower_side_labels)
        higher_side_probs = self.calculate_label_probabilities(higher_side_labels)

        lower_side_entropy = self.calculate_entropy(lower_side_probs)
        higher_side_entropy = self.calculate_entropy(higher_side_probs)

        nodeEntropy = (lower_side_entropy*(len(lower_side_labels) / total_count)) + (higher_side_entropy*(len(higher_side_labels) / total_count))

        return nodeEntropy

    
    def split(self, data, feature_idx, feature_val):
        
        lower_than_threshold = data[data[:, feature_idx] < feature_val]
        higher_than_threshold = data[data[:, feature_idx] >= feature_val]

        return lower_than_threshold, higher_than_threshold
    

    def find_best_split(self, data):
        """
        Finds the best split value and feature
        data: (list) -> list of data
        Returns lower_side, higher_side, split_feature_idx, split_feature_val, split_entropy
        """
        min_part_entropy = 10000000

        for feature_idx in range(len(data[0])-1):
            feature_vals = np.percentile(data[:, feature_idx], q=np.arange(25, 100, 25))
            for feature_val in feature_vals:
                lower_than_threshold, higher_than_threshold, = self.split(data, feature_idx, feature_val)
                part_entropy = self.partition_entropy(lower_than_threshold[:, -1], higher_than_threshold[:, -1])
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = feature_idx
                    min_entropy_feature_val = feature_val
                    lower_side, higher_side = lower_than_threshold, higher_than_threshold

        return lower_side, higher_side, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def create_tree(self, data, current_depth):
        """ 
        Recursive tree creation
        """

        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None
        
        # Find best split
        left_split, right_split, split_feature_idx, split_feature_val, split_entropy = self.find_best_split(data)
        
        # Find label probs for the node
        label_probabilities = self.calculate_label_probabilities(data[:,-1], type='split')

        # Calculate information gain
        node_entropy = self.calculate_entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        
        # Create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > left_split.shape[0] or self.min_samples_leaf > right_split.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node

        current_depth += 1
        node.left = self.create_tree(left_split, current_depth)
        node.right = self.create_tree(right_split, current_depth)
        
        return node
    
    def predict_from_tree(self, X):
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train, Y_train):
        """
        Trains the model with given X and Y datasets
        """

        # Concat features and labels
        self.train_labels = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        # Start creating the tree
        self.tree = self.create_tree(data=train_data, current_depth=0)

    def predict(self, X_set):
        """
        Returns the predicted labels for a given data set
        """

        predictions = [self.predict_from_tree(x) for x in X_set]
        predictions = np.array(predictions)
        preds = np.argmax(predictions, axis=1)
        
        return preds    
        
   

