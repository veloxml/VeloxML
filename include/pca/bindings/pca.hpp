#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "pca/pca.hpp"

namespace py = pybind11;

void bind_pca(py::module_ &m)
{
  py::class_<PCA, UnsupervisedEstimatorBase, std::shared_ptr<PCA>>(m, "PCA")
      .def(py::init<int>())
      .def("fit", &PCA::fit_py)
      .def("transform", &PCA::transform_py)
      .def("predict", &PCA::fit_predict_py)
      .def("fit_transform", &PCA::fit_transform_py)
      .def("get_n_components", &PCA::get_n_components)
      .def("get_mean", &PCA::get_mean)
      .def("get_components", &PCA::get_components)
      .def("check_initialize", &PCA::check_initialize);
}