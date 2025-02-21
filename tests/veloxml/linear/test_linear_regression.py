import pytest
import numpy as np
from veloxml.linear import LinearRegression


@pytest.mark.parametrize("mode", ["LU", "QR", "SVD"])
def test_linear_regression_fit_and_predict(mode):
    """Test training and prediction with different decomposition modes."""
    model = LinearRegression(mode=mode)

    # Generate simple linear data: y = 2x + 1
    X_train = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
    y_train = np.array([3, 5, 7, 9, 11], dtype=np.float64)

    # Train model
    model.fit(X_train, y_train)

    # Check if weights are assigned
    assert model.get_weights() is not None, "Model weights should not be None after training."
    assert model.get_bias() is not None, "Model bias should not be None after training."

    # Predict and check results
    X_test = np.array([[6], [7]], dtype=np.float64)
    y_pred = model.predict(X_test)

    # Expected results: y = 2x + 1
    expected = np.array([13, 15], dtype=np.float64)
    np.testing.assert_almost_equal(y_pred, expected, decimal=5)

@pytest.mark.parametrize("mode", ["LU", "QR", "SVD"])
def test_linear_regression_untrained_predict(mode):
    """Ensure an error is raised if predict is called before fitting."""
    model = LinearRegression(mode=mode)
    
    X_test = np.array([[1], [2]], dtype=np.float64)
    
    with pytest.raises(RuntimeError, match="Model weights are not initialized"):
        model.predict(X_test)
