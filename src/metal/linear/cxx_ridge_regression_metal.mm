#include "metal/linear/cxx_ridge_regression_metal.h"
#include "metal/linear/objcxx_ridge_regression_metal.h"

RidgeRegressionMetal::RidgeRegressionMetal(double lambda) : lambda_(lambda){
    pImpl = [[RidgeRegressionMetalOBJCXX alloc] init];
}

RidgeRegressionMetal::~RidgeRegressionMetal() {
#if !__has_feature(objc_arc)
    [pImpl release];
#endif
}

void RidgeRegressionMetal::fit(const std::vector<double>& X, const std::vector<double>& y, std::size_t rows, std::size_t cols) {
    std::vector<float> X_float(X.begin(), X.end());
    std::vector<float> y_float(y.begin(), y.end());

    [pImpl fitWithX:X_float y:y_float n_samples:rows n_features:cols lambdaVal:lambda_];
}

std::vector<double> RidgeRegressionMetal::predict(const std::vector<double>& X, std::size_t rows, std::size_t cols) {
    std::vector<float> X_float(X.begin(), X.end());
    std::vector<float> y_pred_float = [pImpl predictWithX:X_float rows:rows cols:cols];
    return std::vector<double>(y_pred_float.begin(), y_pred_float.end());
}
