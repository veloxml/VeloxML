#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "kmeans/kmeans.hpp"

namespace py = pybind11;

void bind_kmeans(py::module_ &m)
{
  // KMeansAlgorithm の列挙型を公開
  py::enum_<KMeansAlgorithm>(m, "KMeansAlgorithm")
      .value("STANDARD", KMeansAlgorithm::STANDARD)
      .value("ELKAN", KMeansAlgorithm::ELKAN)
      .value("HAMERLY", KMeansAlgorithm::HAMERLY)
      .export_values();

  py::class_<KMeans, UnsupervisedEstimatorBase, std::shared_ptr<KMeans>>(m, "KMeans")
      .def(py::init<int, int, double, KMeansAlgorithm, bool>(),
           py::arg("n_clusters"), py::arg("max_iter"), py::arg("tol"),
           py::arg("algorithm") = KMeansAlgorithm::STANDARD,
           py::arg("use_kdtree") = false)
      .def("fit", &KMeans::fit_py)
      .def("predict", &KMeans::predict_py)
      .def("transform", &KMeans::transform_py)
      .def("fit_predict", &KMeans::fit_predict_py)
      .def("fit_transform", &KMeans::fit_transform_py)
      .def("get_centroids", &KMeans::get_centroids)
      .def("check_initialize", &KMeans::check_initialize);
}