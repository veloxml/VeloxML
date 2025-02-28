#ifndef CXX_RIDGE_REGRESSION_METAL_H
#define CXX_RIDGE_REGRESSION_METAL_H

#include <vector>
#include "base/regression_base.hpp"


#ifdef __OBJC__
#import "metal/linear/objcxx_ridge_regression_metal.h"
#else
class RidgeRegressionMetalOBJCXX;
#endif

class RidgeRegressionMetal : public RegressionBase {
public:
    RidgeRegressionMetal(double lambda);
    ~RidgeRegressionMetal();

    void fit(const std::vector<double>& X, const std::vector<double>& y, std::size_t rows, std::size_t cols) override;
    std::vector<double> predict(const std::vector<double>& X, std::size_t rows, std::size_t cols) override;

private:
    double lambda_;
    RidgeRegressionMetalOBJCXX* pImpl;
};

#endif // CXX_RIDGE_REGRESSION_METAL_H
