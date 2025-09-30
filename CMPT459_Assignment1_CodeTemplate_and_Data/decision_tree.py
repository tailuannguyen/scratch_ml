# Author: Arash Khoeini
# Email: akhoeini@sfu.ca
# Written for SFU CMPT 459

from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
from node import Node

class DecisionTree(object):
    def __init__(self, criterion: Optional['str'] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None):
        """
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the tree.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.criterion_func = self.entropy if criterion == 'entropy' else self.gini


    def fit(self, X: pd.DataFrame, y: pd.Series)->float:
        """
        :param X: data
        :param y: label column in X
        :return: accuracy of training dataset
        HINT1: You use self.tree to store the root of the tree
        HINT2: You should use self.split_node to split a node
        """
        # Your code
        return self.evaluate(X, y)

    def predict(self, X: pd.DataFrame)->np.ndarray:
        """
        :param X: data
        :return: predict the class for X.
        HINT1: You can use get_child_node method of Node class to traverse
        HINT2: You can use the mode of the class of the leaf node as the prediction
        HINT3: start traverasl from self.tree
        """
        predictions = []
        # Your code
        return np.array(predictions)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> int:
        """
        :param X: data
        :param y: labels
        :return: accuracy of predictions on X
        """
        # preds = self.predict(X)
        # acc = sum(preds == y) / len(preds)
        # return acc
        pass

    def split_node(self, node: Node, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Splits the data in the node into child nodes based on the best feature.

        :param node: the current node to split
        :param X: data in the node
        :param y: labels in the node
        :return: None
        HINT1: Find the best feature to split the data in 'node'.
        HINT2: Use the criterion function (entropy/gini) to score the splits.
        HINT3: Split the data into child nodes
        HINT4: Recursively split the child nodes until the stopping condition is met (e.g., max_depth or single_class).
        """
        # your code
        pass

    def stopping_condition(self, node: Node) -> bool:
        """
        Checks if the stopping condition for splitting is met.

        :param node: The current node
        :return: True if stopping condition is met, False otherwise
        """
        # Check if the node is pure (all labels are the same)
        # Check if the maximum depth is reached
        if (self.min_samples_split is not None and node.size <= self.min_samples_split):
            return True
        if self.max_depth is not None and node.depth >= self.max_depth:
            return True
        if node.single_class:
            return True
        return False
        pass

    def gini(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Returns gini index of the give feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get gini score
        :return:
        """
        # Error handling
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame columns.")

        # Find gini index
        x = X[feature]
        total_sample = len(y)
        gini_index = float('inf')
        if str(x.dtype) == 'object': # Categorical feature
            gini_index = 0.0
            unique_values = x.unique()
            for v in unique_values:
                mask_index = x == v
                y_subset = y[mask_index]
                subset_size = len(y_subset)
                if subset_size == 0: 
                    continue
                class_counts = y_subset.value_counts()
                subset_gini = 1.0 - sum((count / subset_size) ** 2 for count in class_counts)
                gini_index += (subset_size / total_sample) * subset_gini
        else: # Continuous features
            sorted_unique_values = np.sort(x.unique())
            for i in range(len(sorted_unique_values) - 1):
                threshold = (sorted_unique_values[i] + sorted_unique_values[i+1]) / 2
                left_mask_index = x < threshold
                right_mask_index = x >= threshold
                left_subset_size = len(y[left_mask_index])
                right_subset_size = len(y[right_mask_index])
                if left_subset_size == 0 or right_subset_size == 0:
                    continue
                left_subset_gini = 1.0 - sum((count / left_subset_size)**2 for count in y[left_mask_index].value_counts())
                right_subset_gini = 1.0 - sum((count / right_subset_size)**2 for count in y[right_mask_index].value_counts())
                weighted_subset_gini = (left_subset_size / total_sample) * left_subset_gini + (right_subset_size / total_sample) * right_subset_gini
                if weighted_subset_gini < gini_index:
                    gini_index = weighted_subset_gini
        return gini_index if gini_index != float('inf') else 0.0

    def entropy(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Returns entropy of the give feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get entropy score
        :return:
        """
        pass