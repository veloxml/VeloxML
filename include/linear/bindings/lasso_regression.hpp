#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "linear/lasso_regression.hpp"

namespace py = pybind11;

void bind_lasso_regression(py::module_ &m)
{
  py::enum_<LassoSolverMode>(m, "LassoSolverMode")
      .value("FISTA", LassoSolverMode::FISTA)
      .value("ADMM", LassoSolverMode::ADMM)
      .export_values();

  py::class_<LassoRegression, RegressionBase, std::shared_ptr<LassoRegression>>(m, "LassoRegression")
      .def(py::init<double, int, double, LassoSolverMode, double, bool>(),
           py::arg("lambda"), py::arg("max_iter"), py::arg("tol"),
           py::arg("mode") = LassoSolverMode::FISTA,
           py::arg("admm_rho") = 1.0,
           py::arg("penalize_bias") = false)
      // C++ API (vector 入出力)
      .def("fit", &LassoRegression::fit_py)
      .def("predict", &LassoRegression::predict_py)
      .def("get_weights", &LassoRegression::get_weights)
      .def("get_bias", &LassoRegression::get_bias);
}