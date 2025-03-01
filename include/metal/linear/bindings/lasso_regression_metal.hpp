#include "metal/linear/cxx_lasso_regression_metal.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_lasso_regression_metal(py::module_ &m) {
  py::class_<LassoRegressionMetal, RegressionBase,
             std::shared_ptr<LassoRegressionMetal>>(m, "LassoRegressionMetal")
      .def(py::init<double, double, int, double>(), py::arg("lambda"),
           py::arg("lr"), py::arg("max_iter"), py::arg("tol"))
      .def("fit", &LassoRegressionMetal::fit_py)
      .def("predict", &LassoRegressionMetal::predict_py);
}
