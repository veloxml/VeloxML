#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tree/random_forest_classification.hpp"

namespace py = pybind11;

void bind_random_forest_classification(py::module_ &m)
{
  py::class_<RandomForestClassification, ClassificationBase, std::shared_ptr<RandomForestClassification>>(m, "RandomForestClassification")
      .def(py::init<int, int, int, int, double, int, int, Criterion, SplitAlgorithm, int, int, int>(),
           py::arg("n_trees"),
           py::arg("max_depth"),
           py::arg("min_samples_leaf"),
           py::arg("min_samples_split"),
           py::arg("min_impurity_decrease"),
           py::arg("max_leaf_nodes_"),
           py::arg("max_bins"),
           py::arg("tree_mode"),
           py::arg("tree_split_mode"),
           py::arg("max_features"),
           py::arg("n_jobs") = 1,
           py::arg("random_seed") = -1
          )
      .def("fit", &RandomForestClassification::fit_py)
      .def("predict", &RandomForestClassification::predict_py)
      .def("predict_proba", &RandomForestClassification::predict_proba)
      .def("feature_importances", &RandomForestClassification::feature_importances)
      .def("get_n_trees", &RandomForestClassification::get_n_trees)
      .def("get_max_depth", &RandomForestClassification::get_max_depth)
      .def("get_min_samples_leaf", &RandomForestClassification::get_min_samples_leaf)
      .def("get_min_samples_split", &RandomForestClassification::get_min_samples_split)
      .def("get_min_impurity_decrease", &RandomForestClassification::get_min_impurity_decrease)
      .def("get_max_bins", &RandomForestClassification::get_max_bins)
      .def("get_max_features", &RandomForestClassification::get_max_features)
      .def("get_tree_mode", &RandomForestClassification::get_tree_mode)
      .def("get_tree_split_mode", &RandomForestClassification::get_tree_split_mode)
      .def("get_n_jobs", &RandomForestClassification::get_n_jobs)
      .def("check_initialize", &RandomForestClassification::check_initialize);
}