#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tree/decision_tree_classification.hpp"

namespace py = pybind11;

void bind_decision_tree_classification(py::module_ &m)
{
  py::enum_<Criterion>(m, "Criterion")
      .value("Entropy", Criterion::Entropy)
      .value("Gini", Criterion::Gini)
      .value("Logloss", Criterion::Logloss)
      .export_values();

  py::enum_<SplitAlgorithm>(m, "SplitAlgorithm")
      .value("Standard", SplitAlgorithm::Standard)
      .value("Histogram", SplitAlgorithm::Histogram)
      .export_values();

  // DecisionTreeClassification のバインディング
  py::class_<DecisionTreeClassification, ClassificationBase, std::shared_ptr<DecisionTreeClassification>>(m, "DecisionTreeClassification")
      .def(py::init<int, int, int, Criterion, SplitAlgorithm, int, int, double, int>(),
           py::arg("max_depth"),
           py::arg("min_samples_split"),
           py::arg("max_bins"),
           py::arg("criterion") = Criterion::Gini,
           py::arg("split_algorithm") = SplitAlgorithm::Standard,
           py::arg("min_samples_leaf") = 1,
           py::arg("max_leaf_nodes") = 10,
           py::arg("min_impurity_decrease") = 0.0,
           py::arg("max_features") = 5)
      .def("fit", &DecisionTreeClassification::fit_py)
      .def("predict", &DecisionTreeClassification::predict_py)
      .def("feature_importances", &DecisionTreeClassification::compute_feature_importance);
}