import pytest
import numpy as np
from veloxml.tree import DecisionTreeClassification


@pytest.mark.parametrize("criterion, split_algorithm, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, max_features, max_bins", [
    ("Entropy", "Standard", 5, 20, 2, 5, 1.0, 5, 256),
    ("Gini", "Histogram", 10, 10, 1, 8, 0.5, 4, 128),
    ("Logloss", "Standard", 15, 5, 2, 6, 0.1, 4, 256),
    ("Entropy", "Histogram", 3, 30, 5, 4, 0.2, 3, 64),
])
def test_decision_tree_classification_init(criterion, split_algorithm, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, max_features, max_bins):
    """Test initialization with different hyperparameter settings."""
    model = DecisionTreeClassification(
        criterion=criterion,
        split_algorithm=split_algorithm,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        max_features=max_features,
        max_bins=max_bins,
    )

    # assert model.criterion_ == criterion, f"Expected criterion {criterion}, but got {model.criterion_}"
    # assert model.split_algorithm_ == split_algorithm, f"Expected split_algorithm {split_algorithm}, but got {model.split_algorithm_}"
    assert model.max_depth_ == max_depth, f"Expected max_depth {max_depth}, but got {model.max_depth_}"
    assert model.min_samples_split_ == min_samples_split, f"Expected min_samples_split {min_samples_split}, but got {model.min_samples_split_}"
    assert model.min_samples_leaf_ == min_samples_leaf, f"Expected min_samples_leaf {min_samples_leaf}, but got {model.min_samples_leaf_}"
    assert model.max_leaf_nodes_ == max_leaf_nodes, f"Expected max_leaf_nodes {max_leaf_nodes}, but got {model.max_leaf_nodes_}"
    assert model.min_impurity_decrease_ == min_impurity_decrease, f"Expected min_impurity_decrease {min_impurity_decrease}, but got {model.min_impurity_decrease_}"
    assert model.max_features_ == max_features, f"Expected max_features {max_features}, but got {model.max_features_}"
    assert model.max_bins_ == max_bins, f"Expected max_bins {max_bins}, but got {model.max_bins_}"


@pytest.mark.parametrize("criterion, split_algorithm, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, max_features, max_bins", [
    ("Entropy", "Standard", 5, 20, 2, 5, 1.0, 5, 256),
    ("Gini", "Histogram", 10, 10, 1, 8, 0.5, 4, 128),
    ("Logloss", "Standard", 15, 5, 2, 8, 0.1, 3, 256),
    ("Entropy", "Histogram", 3, 30, 5, 4, 0.2, 2, 64),
])
def test_decision_tree_classification_fit_and_predict(criterion, split_algorithm, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, max_features, max_bins):
    """Test training and prediction with different hyperparameter settings."""
    model = DecisionTreeClassification(
        criterion=criterion,
        split_algorithm=split_algorithm,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        max_features=max_features,
        max_bins=max_bins,
    )

    # Generate simple classification data
    X_train = np.array([[0], [1], [2], [3], [4], [5]], dtype=np.float64)
    y_train = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)

    # Train model
    model.fit(X_train, y_train)

    # Check if feature importances are available
    assert model.feature_importances() is not None, "Feature importances should not be None after training."

    # Predict and check results
    X_test = np.array([[6], [7]], dtype=np.float64)
    y_pred = model.predict(X_test)

    # Expected results: y should be 1 for large x
    expected = np.array([1, 1], dtype=np.int32)
    np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("criterion, split_algorithm, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, max_features, max_bins", [
    ("Entropy", "Standard", 5, 20, 2, 5, 1.0, 5, 256),
    ("Gini", "Histogram", 10, 10, 1, 8, 0.5, 4, 128),
    ("Logloss", "Standard", 15, 5, 2, 8, 0.1, 4, 256),
    ("Entropy", "Histogram", 3, 30, 5, 4, 0.2, 3, 64),
])
def test_decision_tree_classification_untrained_predict(criterion, split_algorithm, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, max_features, max_bins):
    """Ensure an error is raised if predict is called before fitting."""
    model = DecisionTreeClassification(
        criterion=criterion,
        split_algorithm=split_algorithm,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        max_features=max_features,
        max_bins=max_bins,
    )

    X_test = np.array([[1], [2]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="Model is not initialized"):
        model.predict(X_test)
