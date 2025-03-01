#ifndef CXX_LASSO_REGRESSION_METAL_H
#define CXX_LASSO_REGRESSION_METAL_H

#include <vector>
#include "base/regression_base.hpp"

// Objective-C++ 環境でのみ、objcxx_xxxxx_yyyyyy_metal.h を読み込む
#ifdef __OBJC__
#import "metal/linear/objcxx_lasso_regression_metal.h"
#else
// C++ の場合は、クラスの前方宣言を行う
class LassoRegressionMetalOBJCXX;
#endif


class LassoRegressionMetal : public RegressionBase {
public:
    LassoRegressionMetal(double lambda, double lr, int max_iter, double tol);
    ~LassoRegressionMetal();

    void fit(const std::vector<double>& X, const std::vector<double>& y, 
             std::size_t rows, std::size_t cols) override;
    std::vector<double> predict(const std::vector<double>& X, std::size_t rows, std::size_t cols) override;

private:
    double lambda_;
    double lr_;
    int max_iter_;
    double tol_;
    LassoRegressionMetalOBJCXX* pImpl;
};

#endif
