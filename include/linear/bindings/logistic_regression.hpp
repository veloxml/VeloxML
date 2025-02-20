#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "linear/logistic_regression.hpp"

namespace py = pybind11;

void bind_logistic_regression(py::module_ &m)
{
  py::enum_<LogisticRegressionSolverType>(m, "LogisticRegressionSolverType")
      .value("LBFGS", LogisticRegressionSolverType::LBFGS)
      .value("NEWTON", LogisticRegressionSolverType::NEWTON)
      .value("CD", LogisticRegressionSolverType::CD)
      .export_values();

  py::class_<LogisticRegression>(m, "LogisticRegression")
      .def(py::init<LogisticRegressionSolverType,
                    double,
                    double,
                    int,
                    double,
                    double,
                    double,
                    int,
                    int>(),
           py::arg("solver") = LogisticRegressionSolverType::LBFGS,
           py::arg("lambda") = 1.0,
           py::arg("tol") = 1e-6,
           py::arg("maxIter") = 100,
           py::arg("ls_alpha_init") = 1.0,
           py::arg("ls_rho") = 0.5,
           py::arg("ls_c") = 1e-4,
           py::arg("history_size") = 10,
           py::arg("lbfgs_k") = 1)
      .def("fit", &LogisticRegression::fit_py)
      .def("predict", &LogisticRegression::predict_py)
      .def("predict_proba", &LogisticRegression::predict_proba_py)
      .def("get_weights", &LogisticRegression::get_weights)
      .def("get_bias", &LogisticRegression::get_bias);
}