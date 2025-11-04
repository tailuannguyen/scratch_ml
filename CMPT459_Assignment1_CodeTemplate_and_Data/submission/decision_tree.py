# Author: Arash Khoeini
# Email: akhoeini@sfu.ca
# Written for SFU CMPT 459

from logging import root
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
        self.tree = Node(
            node_size=len(y),
            node_class=y.mode()[0] if len(y.mode()) > 0 else y.iloc[0],
            depth=0,
            single_class=(len(y.unique()) == 1)
        )
        self.split_node(self.tree, X, y)
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
        
        for idx in X.index:
            sample = X.loc[idx]            
            current_node = self.tree
            prev_node = None

            while not current_node.is_leaf:
                feature_value = sample[current_node.name]
                prev_node = current_node
                current_node = current_node.get_child_node(feature_value)
                if current_node is None:
                    current_node = prev_node
                    break
            
            predictions.append(current_node.node_class)

        return np.array(predictions)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> int:
        """
        :param X: data
        :param y: labels
        :return: accuracy of predictions on X
        """
        preds = self.predict(X)
        acc = sum(preds == y) / len(preds)
        return acc

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

        if self.stopping_condition(node):
            return # Reached stopping condition - node stays as a leaf

        feature_best = None
        impurity_best = float('inf')
        best_threshold = None
        
        for feature in X.columns:
            impurity, threshold = self.criterion_func(X, y, feature)

            if impurity < impurity_best:
                impurity_best = impurity
                feature_best = feature
                best_threshold = threshold

        if feature_best is None:
            return # No valid feature to split on - node stays as a leaf
        
        node.name = feature_best
        x = X[feature_best]

        # Handle categorical feature
        if x.dtype == 'object':
            node.is_numerical = False
            children = {}

            for v in x.unique():
                mask_index = x == v
                X_subset, y_subset = X[mask_index], y[mask_index]

                if len(y_subset) > 0:
                    child_node = Node(node_size=len(y_subset),
                                      node_class=y_subset.mode()[0] if len(y_subset.mode()) > 0 else y_subset.iloc[0],
                                      depth=node.depth + 1,
                                      single_class=(len(y_subset.unique()) == 1)
                    )
                    children[v] = child_node
                    self.split_node(child_node, X_subset.drop(columns=[feature_best]), y_subset)

            if children:
                node.set_children(children)

        # Handle numerical feature
        else:
            node.is_numerical = True
            node.threshold = best_threshold
            left_mask = x < best_threshold
            right_mask = x >= best_threshold
            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[right_mask], y[right_mask]
            children = {}

            if len(y_left) > 0:
                left_node = Node(
                    node_size=len(y_left),
                    node_class=y_left.mode()[0] if len(y_left.mode()) > 0 else y_left.iloc[0],
                    depth=node.depth + 1,
                    single_class=(len(y_left.unique()) == 1)
                )

                children['l'] = left_node
                self.split_node(left_node, X_left, y_left)  # Keep used features, let tree decide whether to split on them again

            if len(y_right) > 0:
                right_node = Node(
                    node_size=len(y_right),
                    node_class=y_right.mode()[0] if len(y_right.mode()) > 0 else y_right.iloc[0],
                    depth=node.depth + 1,
                    single_class=(len(y_right.unique()) == 1)
                )

                children['ge'] = right_node
                self.split_node(right_node, X_right, y_right)   # Keep used features, let tree decide whether to split on them again

            if children:
                node.set_children(children)

    def stopping_condition(self, node: Node) -> bool:
        """
        Checks if the stopping condition for splitting is met.

        :param node: The current node
        :return: True if stopping condition is met, False otherwise
        """
        
        if (self.min_samples_split is not None and node.size <= self.min_samples_split):
            return True
        if self.max_depth is not None and node.depth >= self.max_depth:
            return True
        if node.single_class:
            return True
        if node.size == 0:
            return True
        return False

    def gini(self, X: pd.DataFrame, y: pd.Series, feature: str) -> tuple[float, Optional[float]]:
        """
        Returns gini index of the give feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get gini score
        :return: Tuple of (gini_score, threshold). Threshold is None for categorical features.
        """
        
        x = X[feature]
        sample_size = len(y)

        # handle categorical feature
        if x.dtype == 'object':
            weighted_gini_index = 0.0
            unique_values = x.unique()

            for v in unique_values:
                mask_index = x == v
                y_subset = y[mask_index]
                subset_size = len(y_subset)

                if subset_size == 0: 
                    continue

                subset_gini = 1.0 - sum((count / subset_size) ** 2 for count in y_subset.value_counts())
                weighted_gini_index += (subset_size / sample_size) * subset_gini

            return weighted_gini_index, None

        # handle continuous feature
        else:
            best_gini_index = float('inf')
            best_threshold = None
            sorted_unique_values = np.sort(x.unique())

            for i in range(len(sorted_unique_values) - 1):
                threshold = np.mean([sorted_unique_values[i], sorted_unique_values[i+1]])
                mask_left = x < threshold
                mask_right = x >= threshold
                y_left_subset = y[mask_left]
                y_right_subset = y[mask_right]
                left_subset_size = len(y_left_subset)
                right_subset_size = len(y_right_subset)

                if left_subset_size == 0 or right_subset_size == 0:
                    continue

                left_subset_gini = 1.0 - sum((count / left_subset_size)**2 for count in y_left_subset.value_counts())
                right_subset_gini = 1.0 - sum((count / right_subset_size)**2 for count in y_right_subset.value_counts())
                weighted_gini_index = (left_subset_size / sample_size) * left_subset_gini + (right_subset_size / sample_size) * right_subset_gini

                if weighted_gini_index < best_gini_index:
                    best_gini_index = weighted_gini_index
                    best_threshold = threshold

            best_gini_index = best_gini_index if best_gini_index != float('inf') else None
            return best_gini_index, best_threshold

    def entropy(self, X: pd.DataFrame, y: pd.Series, feature: str) -> tuple[float, Optional[float]]:
        """
        Returns entropy of the give feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get entropy score
        :return: Tuple of (entropy_score, threshold). Threshold is None for categorical features.
        """        
        x = X[feature]
        sample_size = len(y)

        # handle categorical feature
        if x.dtype == 'object':
            weighted_entropy = 0.0
            unique_values = x.unique()

            # calculate sub-entropy for each unique value
            for v in unique_values:
                mask_index = x == v
                y_subset = y[mask_index]
                subset_size = len(y_subset)
                subset_entropy = 0.0

                if subset_size == 0:
                    continue

                for count in y_subset.value_counts():
                    if count > 0:
                        prob = count / subset_size
                        subset_entropy -= prob * np.log2(prob)
            
                weighted_entropy += (subset_size / sample_size) * subset_entropy

            return weighted_entropy, None
        
        # Handle continuous feature
        else:
            best_entropy = float('inf')
            best_threshold = None
            sorted_unique_value = np.sort(x.unique())

            for i in range(len(sorted_unique_value) - 1):
                threshold = (sorted_unique_value[i] + sorted_unique_value[i+1]) / 2
                mask_left = x < threshold
                mask_right = x >= threshold
                left_subset_size = len(y[mask_left])
                right_subset_size = len(y[mask_right])

                if left_subset_size == 0 or right_subset_size == 0:
                    continue

                left_subset_entropy = 0.0
                left_subset_count_normalized = y[mask_left].value_counts(normalize=True)
                left_subset_entropy = -sum(left_subset_count_normalized * np.log2(left_subset_count_normalized + 1e-9))
                right_subset_entropy = 0.0
                right_subset_count_normalized = y[mask_right].value_counts(normalize=True)
                right_subset_entropy = -sum(right_subset_count_normalized * np.log2(right_subset_count_normalized + 1e-9))
                weighted_entropy_candidate = (left_subset_size / sample_size) * left_subset_entropy + (right_subset_size / sample_size) * right_subset_entropy

                if weighted_entropy_candidate < best_entropy:
                    best_entropy = weighted_entropy_candidate
                    best_threshold = threshold

            best_entropy = best_entropy if best_entropy != float('inf') else 0.0
            return best_entropy, best_threshold