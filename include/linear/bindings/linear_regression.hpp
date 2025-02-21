#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "linear/linear_regression.hpp"

namespace py = pybind11;

void bind_linear_regression(py::module_ &m)
{
  py::enum_<LinearDecompositionMode>(m, "LinearDecompositionMode")
        .value("LU", LinearDecompositionMode::LU)
        .value("QR", LinearDecompositionMode::QR)
        .value("SVD", LinearDecompositionMode::SVD)
        .export_values();

    py::class_<LinearRegression, RegressionBase, std::shared_ptr<LinearRegression>>(m, "LinearRegression")
        .def(py::init<LinearDecompositionMode>(), py::arg("mode") = LinearDecompositionMode::LU)
        // C++ API (vector 入出力)
        .def("fit", &LinearRegression::fit_py)
        .def("predict", &LinearRegression::predict_py)
        .def("get_weights", &LinearRegression::get_weights)
        .def("get_bias", &LinearRegression::get_bias);
}