# Copilot Instructions for scratch_ml

## Project Overview
This repository contains machine learning algorithms implemented from scratch (no sklearn). Currently focuses on a decision tree classifier for the CMPT459 course assignment, predicting income levels from census data.

## Architecture & Key Components

### Decision Tree Implementation (`CMPT459_Assignment1_CodeTemplate_and_Data/`)
- **`node.py`**: Tree node structure with dual functionality:
  - Leaf nodes: Store `node_class` (mode of labels) for predictions
  - Decision nodes: Store feature split logic with `is_numerical`, `threshold`, and `children` dict
  - Children keys: `'ge'` (>=) and `'l'` (<) for numerical; actual values for categorical
  
- **`decision_tree.py`**: Core tree logic with incomplete methods requiring implementation:
  - `fit()`, `predict()`, `evaluate()` are stubs - need full implementation
  - `gini()` and `entropy()` are COMPLETE - handle both categorical (weighted impurity per unique value) and continuous features (binary splits at midpoints)
  - `split_node()` needs implementation: find best feature, create children, recurse
  - Stopping conditions already implemented: `max_depth`, `min_samples_split`, `single_class`

- **`main.py`**: Entry point with argparse CLI supporting:
  - `--train-data` / `--test-data` (default: `data/*.csv`)
  - `--criterion` (`gini` or `entropy`)
  - `--maxdepth` (default: 5)
  - `--min-sample-split` (default: 20)
  - NOTE: Missing values handling is a TODO (see "HINT" comment)

- **`test.py`**: Minimal test script for criterion functions only

### Data Structure
- CSV files in `data/` with 14 features + `income` target (<=50K or >50K)
- Mix of categorical (workclass, education, occupation, etc.) and numerical (age, fnlwgt, hours-per-week) features
- Target column: `income` (always drop from X before fitting)

## Development Workflows

### Running the Decision Tree
```bash
cd CMPT459_Assignment1_CodeTemplate_and_Data
python main.py --criterion entropy --maxdepth 5 --min-sample-split 20
```

### Testing Criterion Functions
```bash
cd CMPT459_Assignment1_CodeTemplate_and_Data
python test.py --train-data data/train.csv
```

### Managing Dependencies
```bash
# Project uses uv for package management
uv pip install pandas numpy
# Or add to pyproject.toml dependencies array
```

## Code Conventions

1. **Type Hints**: Use pandas types - `pd.DataFrame`, `pd.Series`, `np.ndarray` - consistently across all methods
2. **Feature Splits**: 
   - Categorical: Create child per unique value
   - Continuous: Binary split at `(sorted_values[i] + sorted_values[i+1]) / 2` thresholds
3. **Node Traversal**: Use `node.get_child_node(feature_value)` - it automatically handles numerical vs categorical logic
4. **Impurity Calculations**: Return weighted average across splits (already implemented in `gini()`/`entropy()`)
5. **Author Attribution**: All course files have "Author: Arash Khoeini" header - preserve when editing

## Critical Implementation Details

- `DecisionTree.fit()` must set `self.tree` to the root Node object
- `Node.single_class` is set during construction, not computed dynamically
- When splitting continuous features, `Node.threshold` must be set and `children` dict must use keys `'ge'` and `'l'`
- `Node.node_class` should be the mode of labels in that node (for leaf predictions)
- Predictions traverse from `self.tree` root to leaf via `get_child_node()`, return leaf's `node_class`

## Files to Ignore
- `__pycache__/` - Python bytecode
- `uv.lock` - Dependency lock file (auto-generated)
- Root-level `main.py` - placeholder unrelated to assignment
