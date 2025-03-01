#include "metal/linear/cxx_lasso_sgd_regression_metal.h"
#include "metal/linear/objcxx_lasso_sgd_regression_metal.h"

#include <iostream>

LassoSGDMetal::LassoSGDMetal(double lambda, double lr, int epochs,
                             int batch_size): lambda_(lambda), lr_(lr), epochs_(epochs), batch_size_(batch_size){
  pImpl = [[LassoSGDMetalOBJCXX alloc] init];
}

LassoSGDMetal::~LassoSGDMetal() {
#if !__has_feature(objc_arc)
  [pImpl release];
#endif
}

void LassoSGDMetal::fit(const std::vector<double> &X,
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
           epochs:epochs_
       batch_size:batch_size_];
}

std::vector<double> LassoSGDMetal::predict(const std::vector<double> &X,
                                           std::size_t rows, std::size_t cols) {
  std::vector<float> X_float(X.begin(), X.end());
  std::vector<float> y_pred_float = [pImpl predictWithX:X_float
                                                   rows:rows
                                                   cols:cols];
  return std::vector<double>(y_pred_float.begin(), y_pred_float.end());
}
