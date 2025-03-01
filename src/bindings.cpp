#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "base/base_binding.hpp"
#include "linear/bindings/linear_regression.hpp"
#include "linear/bindings/ridge_regression.hpp"
#include "linear/bindings/lasso_regression.hpp"
#include "linear/bindings/elasticnet_regression.hpp"
#include "linear/bindings/logistic_regression.hpp"
#include "tree/bindings/decision_tree_classification.hpp"
#include "tree/bindings/random_forest_classification.hpp"
#include "kmeans/bindings/kmeans.hpp"
#include "pca/bindings/pca.hpp"
#include "svm/bindings/svm_classification.hpp"

// Metal Version
#include "metal/linear/bindings/linear_regression_metal.hpp"
#include "metal/linear/bindings/ridge_regression_metal.hpp"
#include "metal/linear/bindings/lasso_regression_metal.hpp"
#include "metal/linear/bindings/lasso_sgd_regression_metal.hpp"

namespace py = pybind11;

PYBIND11_MODULE(c_veloxml_core, m)
{
  m.doc() = "VeloxML core module with multiple ML algorithms";

  bind_estimator_base(m);
  bind_regression_base(m);
  bind_classification_base(m);
  bind_unsupervised_base(m);

  bind_linear_regression(m);
  bind_ridge_regression(m);
  bind_lasso_regression(m);
  bind_elasticnet_regression(m);
  bind_logistic_regression(m);

  bind_decision_tree_classification(m);
  bind_random_forest_classification(m);

  bind_svm_classification(m);

  bind_pca(m);
  bind_kmeans(m);

  // Metal Version
  bind_linear_regression_metal(m);
  bind_ridge_regression_metal(m);
  bind_lasso_regression_metal(m);
  bind_lasso_sgd_regression_metal(m);
}
