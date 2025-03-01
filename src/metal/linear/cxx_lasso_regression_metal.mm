#include "metal/linear/cxx_lasso_regression_metal.h"
#include "metal/linear/objcxx_lasso_regression_metal.h"

#include <iostream>

LassoRegressionMetal::LassoRegressionMetal(double lambda, double lr, int max_iter, double tol)
    : lambda_(lambda), lr_(lr), max_iter_(max_iter), tol_(tol){
  pImpl = [[LassoRegressionMetalOBJCXX alloc] init];
}

LassoRegressionMetal::~LassoRegressionMetal() {
#if !__has_feature(objc_arc)
  [pImpl release];
#endif
}

void LassoRegressionMetal::fit(const std::vector<double> &X,
                               const std::vector<double> &y, std::size_t rows,
                               std::size_t cols) {
  std::vector<float> X_float(X.begin(), X.end());
  std::vector<float> y_float(y.begin(), y.end());

  [pImpl fitWithX:X_float
                y:y_float
             rows:rows
             cols:cols
           lambda:lambda_
               lr:lr_
         max_iter:max_iter_
         tol:tol_];
}

std::vector<double> LassoRegressionMetal::predict(const std::vector<double> &X,
                                                  std::size_t rows,
                                                  std::size_t cols) {
  std::vector<float> X_float(X.begin(), X.end());
  std::vector<float> y_pred_float = [pImpl predictWithX:X_float
                                                   rows:rows
                                                   cols:cols];
  return std::vector<double>(y_pred_float.begin(), y_pred_float.end());
}
