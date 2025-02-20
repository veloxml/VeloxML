#include "base/base_binding.hpp"

void bind_unsupervised_base(py::module_ &m)
{
  py::class_<UnsupervisedEstimatorBase, EstimatorBase, std::shared_ptr<UnsupervisedEstimatorBase>>(m, "UnsupervisedEstimatorBase");
}

void bind_classification_base(py::module_ &m)
{
  py::class_<ClassificationBase, EstimatorBase, std::shared_ptr<ClassificationBase>>(m, "ClassificationBase");
}

void bind_regression_base(py::module_ &m)
{
  py::class_<RegressionBase, EstimatorBase, std::shared_ptr<RegressionBase>>(m, "RegressionBase");
}

void bind_estimator_base(py::module_ &m)
{
  py::class_<EstimatorBase, std::shared_ptr<EstimatorBase>>(m, "EstimatorBase");
}
