#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "linear/ridge_regression.hpp"

namespace py = pybind11;

void bind_ridge_regression(py::module_ &m)
{
  py::class_<RidgeRegression, RegressionBase, std::shared_ptr<RidgeRegression>>(m, "RidgeRegression")
      .def(py::init<double, bool>(),
           py::arg("lambda"), py::arg("penalize_bias") = false)
      // C++ API (vector 入出力)
      .def("fit", &RidgeRegression::fit_py)
      .def("predict", &RidgeRegression::predict_py)
      .def("get_weights", &RidgeRegression::get_weights)
      .def("get_bias", &RidgeRegression::get_bias);
}