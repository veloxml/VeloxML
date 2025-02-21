import pytest
import numpy as np
from veloxml.linear.ridge_regression import RidgeRegression


@pytest.mark.parametrize("lamda_l2, penalize_bias", [(0.1, False), (1.0, True), (10.0, False)])
def test_ridge_regression_init(lamda_l2, penalize_bias):
    """Test initialization with different lambda values and bias penalization settings."""
    model = RidgeRegression(lamda_l2=lamda_l2, penalize_bias=penalize_bias)

    assert model.lamda_l2_ == lamda_l2, f"Expected lamda_l2 {lamda_l2}, but got {model.lamda_l2_}"
    assert model.penalize_bias_ == penalize_bias, f"Expected penalize_bias {penalize_bias}, but got {model.penalize_bias_}"


@pytest.mark.parametrize("lamda_l2, penalize_bias", [(0.1, False), (1.0, True), (10.0, False)])
def test_ridge_regression_fit_and_predict(lamda_l2, penalize_bias):
    """Test training and prediction with different lambda values and bias penalization settings."""
    model = RidgeRegression(lamda_l2=lamda_l2, penalize_bias=penalize_bias)

    # Generate simple linear data: y = 3x + 2
    X_train = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
    y_train = np.array([5, 8, 11, 14, 17], dtype=np.float64)

    # Train model
    model.fit(X_train, y_train)

    # Check if weights are assigned
    assert model.get_weights() is not None, "Model weights should not be None after training."
    assert model.get_bias() is not None, "Model bias should not be None after training."

    # Predict and check results
    X_test = np.array([[6], [7]], dtype=np.float64)
    y_pred = model.predict(X_test)

    # Expected results: y = 3x + 2
    expected = np.array([20, 23], dtype=np.float64)
    np.testing.assert_almost_equal(y_pred, expected, decimal=-2)


@pytest.mark.parametrize("lamda_l2, penalize_bias", [(0.1, False), (1.0, True), (10.0, False)])
def test_ridge_regression_untrained_predict(lamda_l2, penalize_bias):
    """Ensure an error is raised if predict is called before fitting."""
    model = RidgeRegression(lamda_l2=lamda_l2, penalize_bias=penalize_bias)

    X_test = np.array([[1], [2]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="Model weights are not initialized"):
        model.predict(X_test)
