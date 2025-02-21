#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "linear/elasticnet_regression.hpp"

namespace py = pybind11;

void bind_elasticnet_regression(py::module_ &m)
{
     py::enum_<ElasticNetSolverMode>(m, "ElasticNetSolverMode")
         .value("FISTA", ElasticNetSolverMode::FISTA)
         .value("ADMM", ElasticNetSolverMode::ADMM)
         .export_values();

     py::class_<ElasticnetRegression, RegressionBase, std::shared_ptr<ElasticnetRegression>>(m, "ElasticnetRegression")
         .def(py::init<double, double, int, double, ElasticNetSolverMode, double, bool>(),
              py::arg("lambda1"), py::arg("lambda2"), py::arg("max_iter"), py::arg("tol"),
              py::arg("mode") = ElasticNetSolverMode::FISTA,
              py::arg("admm_rho") = 1.0,
              py::arg("penalize_bias") = false)
         // C++ API (vector 入出力)
         .def("fit", &ElasticnetRegression::fit_py)
         .def("predict", &ElasticnetRegression::predict_py)
         .def("get_weights", &ElasticnetRegression::get_weights)
         .def("get_bias", &ElasticnetRegression::get_bias);
}