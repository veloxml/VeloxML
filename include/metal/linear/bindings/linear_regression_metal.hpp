#include "metal/linear/cxx_linear_regression_metal.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_linear_regression_metal(py::module_ &m) {
  py::class_<LinearRegressionMetal, RegressionBase, std::shared_ptr<LinearRegressionMetal>>(
      m, "LinearRegressionMetal")
      .def(py::init<>())
      .def("fit", &LinearRegressionMetal::fit_py)
      .def("predict", &LinearRegressionMetal::predict_py);
}
