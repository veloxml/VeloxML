#ifndef CXX_LINEAR_REGRESSION_METAL_H
#define CXX_LINEAR_REGRESSION_METAL_H

#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


// Objective-C++ 環境でのみ、objcxx_linear_regression_metal.h を読み込む
#ifdef __OBJC__
#import "metal/linear/objcxx_linear_regression_metal.h"
#else
// C++ の場合は、クラスの前方宣言を行う
class LinearRegressionMetalOBJCXX;
#endif

// ARC の有無に応じた __strong マクロ
#if defined(__OBJC__)
  #if __has_feature(objc_arc)
    #define OBJC_STRONG __strong
  #else
    #define OBJC_STRONG
  #endif
#else
  #define OBJC_STRONG
#endif

class LinearRegressionMetal {
public:
    LinearRegressionMetal();
    ~LinearRegressionMetal();

    void fit(const std::vector<double>& X, const std::vector<double>& y, std::size_t rows, std::size_t cols);
    std::vector<double> predict(const std::vector<double>& X, std::size_t rows, std::size_t cols);

    void fit_py(const py::array_t<double> &X, const py::array_t<double> &Y)
  {
    auto bufX = X.request();
    auto bufY = Y.request();

    if (bufX.ndim != 2)
      throw std::runtime_error("fit: X must be a 2-dimensional array");

    std::size_t n = bufX.shape[0];
    std::size_t m = bufX.shape[1];

    std::vector<double> vecX(static_cast<double *>(bufX.ptr), static_cast<double *>(bufX.ptr) + n * m);
    std::vector<double> vecY(static_cast<double *>(bufY.ptr), static_cast<double *>(bufY.ptr) + n);

    fit(vecX, vecY, n, m);
  }

  py::array_t<double> predict_py(const py::array_t<double> &X)
  {
    auto bufX = X.request();
    if (bufX.ndim != 2)
      throw std::runtime_error("predict: X must be a 2-dimensional array");

    std::size_t l = bufX.shape[0];
    std::size_t m = bufX.shape[1];

    std::vector<double> vecX(static_cast<double *>(bufX.ptr), static_cast<double *>(bufX.ptr) + l * m);
    std::vector<double> result = predict(vecX, l, m);

    return py::array_t<double>(result.size(), result.data());
  }

private:
    OBJC_STRONG LinearRegressionMetalOBJCXX* pImpl;
};

#endif // CXX_LINEAR_REGRESSION_METAL_H
