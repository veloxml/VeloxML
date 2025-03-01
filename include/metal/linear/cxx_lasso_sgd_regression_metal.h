#ifndef CXX_LASSO_SGD_METAL_H
#define CXX_LASSO_SGD_METAL_H

#include <vector>
#include "base/regression_base.hpp"

// Objective-C++ 環境でのみ、objcxx_xxxxx_yyyyyy_metal.h を読み込む
#ifdef __OBJC__
#import "metal/linear/objcxx_lasso_sgd_regression_metal.h"
#else
// C++ の場合は、クラスの前方宣言を行う
class LassoSGDMetalOBJCXX;
#endif

class LassoSGDMetal : public RegressionBase{
public:
  LassoSGDMetal(double lambda, double lr, int epochs, int batch_size);
  ~LassoSGDMetal();

  void fit(const std::vector<double> &X, const std::vector<double> &y,
           std::size_t rows, std::size_t cols) override;
  std::vector<double> predict(const std::vector<double> &X, std::size_t rows,
                              std::size_t cols) override;

private:
  double lambda_;
  double lr_;
  int epochs_;
  int batch_size_;
  LassoSGDMetalOBJCXX *pImpl;
};

#endif // CXX_LASSO_SGD_METAL_H
