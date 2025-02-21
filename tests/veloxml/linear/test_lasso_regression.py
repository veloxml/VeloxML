import pytest
import numpy as np
from veloxml.linear.lasso_regression import LassoRegression


@pytest.mark.parametrize("mode, lambda_l1, max_iter, tol, admm_rho, penalize_bias", [
    ("FISTA", 0.1, 100, 1e-4, 1.0, False),
    ("FISTA", 1.0, 200, 1e-5, 0.5, True),
    ("ADMM", 0.5, 150, 1e-3, 2.0, False),
    ("ADMM", 10.0, 300, 1e-6, 0.1, True),
])
def test_lasso_regression_init(mode, lambda_l1, max_iter, tol, admm_rho, penalize_bias):
    """Test initialization with different hyperparameter settings."""
    model = LassoRegression(
        mode=mode,
        lambda_l1=lambda_l1,
        max_iter=max_iter,
        tol=tol,
        admm_rho=admm_rho,
        penalize_bias=penalize_bias,
    )

    assert model.lambda_l1_ == lambda_l1, f"Expected lambda_l1 {lambda_l1}, but got {model.lambda_l1_}"
    assert model.max_iter_ == max_iter, f"Expected max_iter {max_iter}, but got {model.max_iter_}"
    assert model.tol_ == tol, f"Expected tol {tol}, but got {model.tol_}"
    assert model.admm_rho_ == admm_rho, f"Expected admm_rho {admm_rho}, but got {model.admm_rho_}"
    assert model.penalize_bias_ == penalize_bias, f"Expected penalize_bias {penalize_bias}, but got {model.penalize_bias_}"
    # assert model.solve_mode == mode, f"Expected mode {mode}, but got {model.solve_mode}"


@pytest.mark.parametrize("mode, lambda_l1, max_iter, tol, admm_rho, penalize_bias", [
    ("FISTA", 0.1, 100, 1e-4, 1.0, False),
    ("FISTA", 1.0, 200, 1e-5, 0.5, True),
    ("ADMM", 0.5, 150, 1e-3, 2.0, False),
    ("ADMM", 10.0, 300, 1e-6, 0.1, True),
])
def test_lasso_regression_fit_and_predict(mode, lambda_l1, max_iter, tol, admm_rho, penalize_bias):
    """Test training and prediction with different hyperparameter settings."""
    model = LassoRegression(
        mode=mode,
        lambda_l1=lambda_l1,
        max_iter=max_iter,
        tol=tol,
        admm_rho=admm_rho,
        penalize_bias=penalize_bias,
    )

    # Generate simple linear data: y = 2x + 3
    X_train = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
    y_train = np.array([5, 7, 9, 11, 13], dtype=np.float64)

    # Train model
    model.fit(X_train, y_train)

    # Check if weights are assigned
    assert model.get_weights() is not None, "Model weights should not be None after training."
    assert model.get_bias() is not None, "Model bias should not be None after training."

    # Predict and check results
    X_test = np.array([[6], [7]], dtype=np.float64)
    y_pred = model.predict(X_test)

    # Expected results: y = 2x + 3
    expected = np.array([15, 17], dtype=np.float64)
    np.testing.assert_almost_equal(y_pred, expected, decimal=0)


@pytest.mark.parametrize("mode, lambda_l1, max_iter, tol, admm_rho, penalize_bias", [
    ("FISTA", 0.1, 100, 1e-4, 1.0, False),
    ("FISTA", 1.0, 200, 1e-5, 0.5, True),
    ("ADMM", 0.5, 150, 1e-3, 2.0, False),
    ("ADMM", 10.0, 300, 1e-6, 0.1, True),
])
def test_lasso_regression_untrained_predict(mode, lambda_l1, max_iter, tol, admm_rho, penalize_bias):
    """Ensure an error is raised if predict is called before fitting."""
    model = LassoRegression(
        mode=mode,
        lambda_l1=lambda_l1,
        max_iter=max_iter,
        tol=tol,
        admm_rho=admm_rho,
        penalize_bias=penalize_bias,
    )

    X_test = np.array([[1], [2]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="Model weights are not initialized"):
        model.predict(X_test)
