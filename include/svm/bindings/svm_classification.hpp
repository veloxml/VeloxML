#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "svm/svm_classification.hpp"

namespace py = pybind11;

void bind_svm_classification(py::module_ &m)
{
  py::class_<SVMClassification, ClassificationBase, std::shared_ptr<SVMClassification>>(m, "SVMClassification")
      .def(py::init<double, double, int, const std::string &, bool, double, double, int, bool>(),
           py::arg("C"),
           py::arg("tol"),
           py::arg("max_passes"),
           py::arg("kernel"),
           py::arg("gamma_scale") = true,
           py::arg("gamma") = 0.1,
           py::arg("coef0") = 0.0,
           py::arg("degree") = 3,
           py::arg("approx_kernel") = false)
      .def("fit", &SVMClassification::fit_py)
      .def("predict", &SVMClassification::predict_py)
      .def("predict_proba", &SVMClassification::predict_proba_py)
      .def("predict_score", [](SVMClassification &self, py::array_t<double> X)
           {
      auto buf = X.request();
      if (buf.ndim != 2)
          throw std::runtime_error("X must be 2-dimensional");
      std::size_t l = buf.shape[0];
      std::size_t m = buf.shape[1];
      std::vector<double> vecX(static_cast<double*>(buf.ptr), static_cast<double*>(buf.ptr) + l * m);
      std::vector<double> result = self.predict_score(vecX, l, m);
      py::array_t<double>({static_cast<py::ssize_t>(l)}, result.data()); })
      // ゲッター
      .def("getC", &SVMClassification::getC)
      .def("getTol", &SVMClassification::getTol)
      .def("getMaxPasses", &SVMClassification::getMaxPasses)
      .def("getKernel", &SVMClassification::getKernel)
      .def("getGamma", &SVMClassification::getGamma)
      .def("getCoef0", &SVMClassification::getCoef0)
      .def("getDegree", &SVMClassification::getDegree)
      .def("getApproxKernel", &SVMClassification::getApproxKernel)
      .def("check_initialize", &SVMClassification::check_initialize);
}