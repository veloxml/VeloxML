#include "metal/linear/cxx_lasso_sgd_regression_metal.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_lasso_sgd_regression_metal(py::module_ &m) {
  py::class_<LassoSGDMetal, RegressionBase,
             std::shared_ptr<LassoSGDMetal>>(m, "LassoSGDMetal")
      .def(py::init<double, double, int, int>(), py::arg("lambda"),
           py::arg("lr"), py::arg("epochs"), py::arg("batch_size"))
      .def("fit", &LassoSGDMetal::fit_py)
      .def("predict", &LassoSGDMetal::predict_py);
}
