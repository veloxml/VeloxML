import pytest
import numpy as np
from veloxml.linear import LogisticRegression


@pytest.mark.parametrize("solver_type, lamda_l2, tol, max_iter, ls_alpha_init, ls_rho, ls_c, history_size", [
    ("LBFGS", 0.1, 1e-6, 100, 1.0, 0.5, 1e-4, 10),
    ("LBFGS", 1.0, 1e-5, 200, 0.5, 0.8, 1e-3, 15),
    ("NEWTONCG", 0.5, 1e-4, 150, 2.0, 0.3, 1e-5, 20),
    ("NEWTONCG", 10.0, 1e-6, 300, 1.5, 0.6, 1e-2, 5),
])
def test_logistic_regression_init(solver_type, lamda_l2, tol, max_iter, ls_alpha_init, ls_rho, ls_c, history_size):
    """Test initialization with different hyperparameter settings."""
    model = LogisticRegression(
        solver_type=solver_type,
        lamda_l2=lamda_l2,
        tol=tol,
        max_iter=max_iter,
        ls_alpha_init=ls_alpha_init,
        ls_rho=ls_rho,
        ls_c=ls_c,
        history_size=history_size,
    )

    assert model.lamda_l2_ == lamda_l2, f"Expected lamda_l2 {lamda_l2}, but got {model.lamda_l2_}"
    assert model.tol_ == tol, f"Expected tol {tol}, but got {model.tol_}"
    assert model.max_iter_ == max_iter, f"Expected max_iter {max_iter}, but got {model.max_iter_}"
    assert model.ls_alpha_init_ == ls_alpha_init, f"Expected ls_alpha_init {ls_alpha_init}, but got {model.ls_alpha_init_}"
    assert model.ls_rho_ == ls_rho, f"Expected ls_rho {ls_rho}, but got {model.ls_rho_}"
    assert model.ls_c_ == ls_c, f"Expected ls_c {ls_c}, but got {model.ls_c_}"
    assert model.history_size_ == history_size, f"Expected history_size {history_size}, but got {model.history_size_}"
    # assert model.solver_type == solver_type, f"Expected solver_type {solver_type}, but got {model.solver_type}"


@pytest.mark.parametrize("solver_type, lamda_l2, tol, max_iter, ls_alpha_init, ls_rho, ls_c, history_size", [
    ("LBFGS", 0.1, 1e-6, 100, 1.0, 0.5, 1e-4, 10),
    ("LBFGS", 1.0, 1e-5, 200, 0.5, 0.8, 1e-3, 15),
    ("NEWTONCG", 0.5, 1e-4, 150, 2.0, 0.3, 1e-5, 20),
    ("NEWTONCG", 10.0, 1e-6, 300, 1.5, 0.6, 1e-2, 5),
])
def test_logistic_regression_fit_and_predict(solver_type, lamda_l2, tol, max_iter, ls_alpha_init, ls_rho, ls_c, history_size):
    """Test training and prediction with different hyperparameter settings."""
    model = LogisticRegression(
        solver_type=solver_type,
        lamda_l2=lamda_l2,
        tol=tol,
        max_iter=max_iter,
        ls_alpha_init=ls_alpha_init,
        ls_rho=ls_rho,
        ls_c=ls_c,
        history_size=history_size,
    )

    # Generate simple classification data
    X_train = np.array([[0], [1], [2], [3], [4], [5]], dtype=np.float64)
    y_train = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)

    # Train model
    model.fit(X_train, y_train)

    # Check if weights are assigned
    assert model.get_weights() is not None, "Model weights should not be None after training."
    assert model.get_bias() is not None, "Model bias should not be None after training."

    # Predict and check results
    X_test = np.array([[6], [7]], dtype=np.float64)
    y_pred = model.predict(X_test)

    # Expected results: y should be 1 for large x
    expected = np.array([1, 1], dtype=np.int32)
    np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("solver_type, lamda_l2, tol, max_iter, ls_alpha_init, ls_rho, ls_c, history_size", [
    ("LBFGS", 0.1, 1e-6, 100, 1.0, 0.5, 1e-4, 10),
    ("LBFGS", 1.0, 1e-5, 200, 0.5, 0.8, 1e-3, 15),
    ("NEWTONCG", 0.5, 1e-4, 150, 2.0, 0.3, 1e-5, 20),
    ("NEWTONCG", 10.0, 1e-6, 300, 1.5, 0.6, 1e-2, 5),
])
def test_logistic_regression_untrained_predict(solver_type, lamda_l2, tol, max_iter, ls_alpha_init, ls_rho, ls_c, history_size):
    """Ensure an error is raised if predict is called before fitting."""
    model = LogisticRegression(
        solver_type=solver_type,
        lamda_l2=lamda_l2,
        tol=tol,
        max_iter=max_iter,
        ls_alpha_init=ls_alpha_init,
        ls_rho=ls_rho,
        ls_c=ls_c,
        history_size=history_size,
    )

    X_test = np.array([[1], [2]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="Model weights are not initialized"):
        model.predict(X_test)
