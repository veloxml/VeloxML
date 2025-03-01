#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "svm/svm_regression.hpp"

namespace py = pybind11;

void bind_svm_regression(py::module_ &m)
{
  py::enum_<KernelType>(m, "SVMRegressionKernelType")
    .value("LINEAR", KernelType::LINEAR)
    .value("POLYNOMIAL", KernelType::POLYNOMIAL)
    .value("RBF", KernelType::RBF)
    .value("APPROX_RBF", KernelType::APPROX_RBF);

  py::class_<SVMRegression, RegressionBase, std::shared_ptr<SVMRegression>>(m, "SVMRegression")
      .def(py::init<double, double, double, int, KernelType, double, int, double, int>(),
           py::arg("C"),
           py::arg("epsilon"),
           py::arg("tol"),
           py::arg("max_iter"),
           py::arg("kernel_type"),
           py::arg("gamma"),
           py::arg("degree") = 0.1,
           py::arg("coef0") = 0.0,
           py::arg("approx_dim") = 3)
      .def("fit", &SVMRegression::fit_py)
      .def("predict", &SVMRegression::predict_py);
      // ゲッター
      // .def("getC", &SVMRegression::getC)
      // .def("getTol", &SVMRegression::getTol)
      // .def("getMaxPasses", &SVMRegression::getMaxPasses)
      // .def("getKernel", &SVMRegression::getKernel)
      // .def("getGamma", &SVMRegression::getGamma)
      // .def("getCoef0", &SVMRegression::getCoef0)
      // .def("getDegree", &SVMRegression::getDegree)
      // .def("getApproxKernel", &SVMRegression::getApproxKernel)
      // .def("check_initialize", &SVMRegression::check_initialize);
}