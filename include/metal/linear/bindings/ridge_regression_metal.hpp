#include "metal/linear/cxx_ridge_regression_metal.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_ridge_regression_metal(py::module_ &m) {
  py::class_<RidgeRegressionMetal, RegressionBase, std::shared_ptr<RidgeRegressionMetal>>(
      m, "RidgeRegressionMetal")
      .def(py::init<double>(), py::arg("lambda"))
      .def("fit", &RidgeRegressionMetal::fit_py)
      .def("predict", &RidgeRegressionMetal::predict_py);
}
