# Author: Arash Khoeini
# Email: akhoeini@sfu.ca
# Written for SFU CMPT 459

import pandas as pd
import argparse
from decision_tree import DecisionTree
from typing import Tuple, Dict

def parse_args() -> Dict[str, any]:
    """
    Parses command-line arguments.

    :return: Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run decision tree with specified input arguments')
    parser.add_argument('--train-data', type=str, default='data/train.csv',
                        help='Path to the training data')
    parser.add_argument('--test-data', type=str, default='data/test.csv',
                        help='Path to the testing data')
    parser.add_argument('--criterion', type=str, default='entropy',
                        help='Criterion to use for splitting nodes. Should be either "gini" or "entropy".')
    parser.add_argument('--maxdepth', type=int, default=5,
                        help='Maximum depth of the tree')
    parser.add_argument('--min-sample-split', type=int, default=20,
                        help='Minimum number of samples required to split a node')
    args = parser.parse_args()
    return vars(args)

def read_data(path: str) -> pd.DataFrame:
    """
    Reads data from a CSV file.

    :param path: Path to the CSV file
    :return: DataFrame containing the data
    """
    data = pd.read_csv(path)
    return data

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    for col in data.columns:
        if data[col].dtype == 'object':
            mode = data[col].mode()[0]
            data[col] = data[col].replace('?', mode)
        else:
            mean = data[col].mean()
            data[col] = data[col].replace('?', mean)
    return data

def main() -> None:
    """
    Main function to run the decision tree algorithm.
    """
    args = parse_args()
    train_data = read_data(args['train_data'])
    test_data = read_data(args['test_data'])

    # Handle missing values in the data
    # HINT: You can drop rows with missing values, or better, replace their values with the mean of the column

    # Replace missing values with mean for numerical columns and mode for categorical columns
    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)
    
    dt = DecisionTree(criterion=args['criterion'],
                      max_depth=args['maxdepth'],
                      min_samples_split=args['min_sample_split'])

    X_train = train_data.drop('income', axis=1)
    y_train = train_data['income']
    X_test = test_data.drop('income', axis=1)
    y_test = test_data['income']

    dt.fit(X_train, y_train)
    print("Training Accuracy:", dt.evaluate(X_train, y_train))
    print("Testing Accuracy:", dt.evaluate(X_test, y_test))

if __name__ == '__main__':
    main()