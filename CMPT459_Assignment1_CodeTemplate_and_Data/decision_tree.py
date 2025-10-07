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

        # error handling
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame columns.")

        # find gini index
        x = X[feature]
        sample_size = len(y)

        # handle categorical feature
        if x.dtype == 'object':

            weighted_gini_index = 0.0
            unique_values = x.unique()

            # calculate sub-gini for each unique value
            for v in unique_values:

                mask_index = x == v
                subset_y = y[mask_index]
                subset_size = len(subset_y)

                # nothing to calculate - skip value
                if subset_size == 0: 
                    continue

                subset_gini = 1.0 - sum((count / subset_size) ** 2 for count in subset_y.value_counts())
                weighted_gini_index += (subset_size / sample_size) * subset_gini

            return weighted_gini_index

        # handle continuous feature
        else:

            best_gini_index = float('inf')
            sorted_unique_values = np.sort(x.unique())

            # try splitting at each threshold
            for i in range(len(sorted_unique_values) - 1):

                threshold = (sorted_unique_values[i] + sorted_unique_values[i+1]) / 2
                left_mask_index = x < threshold
                right_mask_index = x >= threshold
                left_subset_y = y[left_mask_index]
                right_subset_y = y[right_mask_index]
                left_subset_size = len(left_subset_y)
                right_subset_size = len(right_subset_y)

                # cannot split - skip threshold
                if left_subset_size == 0 or right_subset_size == 0:
                    continue

                # calculate gini index for the split
                left_subset_gini = 1.0 - sum((count / left_subset_size)**2 for count in left_subset_y.value_counts())
                right_subset_gini = 1.0 - sum((count / right_subset_size)**2 for count in right_subset_y.value_counts())

                # calculate weighted gini
                weighted_gini_index_candidate = (left_subset_size / sample_size) * left_subset_gini + (right_subset_size / sample_size) * right_subset_gini
                if weighted_gini_index_candidate < best_gini_index:
                    best_gini_index = weighted_gini_index_candidate
        
    def entropy(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Returns entropy of the give feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get entropy score
        :return:
        """

        # Input check
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame columns.")
        
        x = X[feature]
        sample_size = len(y)

        # handle categorical feature
        if x.dtype == 'object':

            weighted_entropy = 0.0
            unique_values = x.unique()

            # calculate sub-entropy for each unique value
            for v in unique_values:

                mask_index = x == v
                subset_y = y[mask_index]
                subset_size = len(subset_y)
                subset_entropy = 0.0

                # nothing to calculate - skip value
                if subset_size == 0:
                    continue

                # calculate value's entropy
                for count in subset_y.value_counts():
                    if count > 0:
                        prob = count / subset_size
                        subset_entropy -= prob * np.log2(prob)
            
                # calculate weighted entropy
                weighted_entropy += (subset_size / sample_size) * subset_entropy

            return weighted_entropy
        
        # Handle continuous feature
        else:

            best_entropy = float('inf')
            sorted_unique_value = np.sort(x.unique())

            # try splitting at each threshold
            for i in range(len(sorted_unique_value) - 1):

                threshold = (sorted_unique_value[i] + sorted_unique_value[i+1]) / 2
                left_mask_index = x < threshold
                right_mask_index = x >= threshold
                left_subset_size = len(y[left_mask_index])
                right_subset_size = len(y[right_mask_index])

                # cannot split - skip threshold
                if left_subset_size == 0 or right_subset_size == 0:
                    continue

                # calculate left entropy
                left_subset_entropy = 0.0
                for count in y[left_mask_index].value_counts():
                    if count > 0:
                        prob = count / left_subset_size
                        left_subset_entropy -= prob * np.log2(prob)

                # calculate right entropy
                right_subset_entropy = 0.0
                for count in y[right_mask_index].value_counts():
                    if count > 0:
                        prob = count / right_subset_size
                        right_subset_entropy -= prob * np.log2(prob)

                # calculate weighted entropy
                weighted_entropy_candidate = (left_subset_size / sample_size) * left_subset_entropy + (right_subset_size / sample_size) * right_subset_entropy
                if weighted_entropy_candidate < best_entropy:
                    best_entropy = weighted_entropy_candidate
                    # best_threshold = threshold

            return best_entropy if best_entropy != float('inf') else 0.0